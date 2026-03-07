# Copyright (c) 2025-2026, NVIDIA CORPORATION. All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

import ctypes
import math
import sys
from typing import Any, List, Optional

from . import bindings
from .dlpack import DLDataType, DLDataTypeCode, DLDevice, DLDeviceType, DLTensor
from .types import (
    AttributeBinding,
    AttributeMapping,
    DataAccess,
    Device,
    FrameOutput,
    Operation,
    PrimMode,
    ProductOutput,
    RendererConfig,
    RendererResult,
    RenderProductSetOutputs,
    RenderVarOutput,
    Semantic,
)


class _AttributeBindingDescStorage:
    """Helper class to keep binding descriptor and string arrays alive (private implementation detail)."""

    def __init__(
        self,
        prim_paths: List[str],
        attribute_name: str,
        dtype: DLDataType,
        semantic: int,
        prim_mode: PrimMode = PrimMode.EXISTING_ONLY,
        flags: int = bindings.OVRTX_BINDING_FLAG_NONE,
        is_array: bool = False,
    ):
        # Keep string objects alive
        self._prim_strings = [bindings.ovx_string_t(path) for path in prim_paths]
        self._attr_string = bindings.ovx_string_t(attribute_name)

        # Create ctypes array - MUST keep reference to prevent GC
        self._prim_array = (bindings.ovx_string_t * len(self._prim_strings))(*self._prim_strings)

        # Build prim_list pointing to array
        self.prim_list = bindings.ovrtx_prim_list_t(
            prim_paths=self._prim_array, num_paths=len(self._prim_strings)  # Array reference kept in self
        )

        # Build attribute_name (string mode)
        self.attribute_name = bindings.ovx_string_or_token_t(
            string=self._attr_string, token=0  # String reference kept in self
        )

        # Build attribute_type
        self.attribute_type = bindings.ovrtx_attribute_type_t(
            dtype=dtype,
            is_array=is_array,
            semantic=bindings.ovrtx_attribute_semantic_t(semantic),
        )

        # Build full binding_desc
        self.binding_desc = bindings.ovrtx_binding_desc_t(
            prim_list=self.prim_list,
            prims_list_handle=0,  # Not using persistent handles
            attribute_name=self.attribute_name,
            attribute_type=self.attribute_type,
            prim_mode=bindings.ovrtx_binding_prim_mode_t(prim_mode),
            flags=bindings.ovrtx_binding_flag_t(flags),
        )

        # Build binding_desc_or_handle (always use binding_desc, handle=0)
        self.binding_desc_or_handle = bindings.ovrtx_binding_desc_or_handle_t(
            binding_desc=self.binding_desc, binding_handle=0
        )


class _InputBufferStorage:
    """Helper class to keep input buffer and DLTensor arrays alive (private implementation detail)."""

    def __init__(
        self,
        dl_tensors: List[DLTensor],
        dirty_bits: Optional[bytes] = None,
        cuda_stream: Optional[int] = None,
        cuda_event: Optional[int] = None,
    ):
        """Initialize input buffer storage.

        Args:
            dl_tensors: List of DLTensor objects (will be copied to array)
            dirty_bits: Optional dirty bit array (copied if provided)
            cuda_stream: Optional CUDA stream handle (int). Sets both access and done sync stream fields.
            cuda_event: Optional CUDA event handle (int). Sets access sync wait_event field.
        """
        # Keep Python data references alive (for ASYNC access)
        self._tensor_refs = dl_tensors

        # Create DLTensor objects array
        self._tensor_storage = list(dl_tensors)  # Keep references

        # Create ctypes array and populate with stable shape/strides pointers.
        # We store the backing arrays in _shape_storage/_strides_storage to keep them alive.
        tensor_count = len(self._tensor_storage)
        self._tensor_array = (DLTensor * tensor_count)()
        self._shape_storage = []
        self._strides_storage = []

        for i, src_tensor in enumerate(self._tensor_storage):
            dest = self._tensor_array[i]
            # Copy scalar fields
            dest.data = src_tensor.data
            dest.device = src_tensor.device
            dest.ndim = src_tensor.ndim
            dest.dtype = src_tensor.dtype
            dest.byte_offset = src_tensor.byte_offset

            # Deep-copy shape array and set pointer
            if src_tensor.ndim > 0 and src_tensor.shape:
                shape_array = (ctypes.c_int64 * src_tensor.ndim)()
                for j in range(src_tensor.ndim):
                    shape_array[j] = src_tensor.shape[j]
                self._shape_storage.append(shape_array)
                dest.shape = ctypes.cast(shape_array, ctypes.POINTER(ctypes.c_int64))
            else:
                self._shape_storage.append(None)
                dest.shape = None

            # Deep-copy strides array if present
            if src_tensor.ndim > 0 and src_tensor.strides:
                strides_array = (ctypes.c_int64 * src_tensor.ndim)()
                for j in range(src_tensor.ndim):
                    strides_array[j] = src_tensor.strides[j]
                self._strides_storage.append(strides_array)
                dest.strides = ctypes.cast(strides_array, ctypes.POINTER(ctypes.c_int64))
            else:
                self._strides_storage.append(None)
                dest.strides = None

        # Build dirty_bits array if provided
        self._dirty_bits_array = None
        dirty_bits_ptr = None
        dirty_bits_size = 0
        if dirty_bits is not None:
            dirty_bits_size = len(dirty_bits)
            self._dirty_bits_array = (ctypes.c_uint8 * dirty_bits_size).from_buffer_copy(dirty_bits)
            dirty_bits_ptr = ctypes.cast(self._dirty_bits_array, ctypes.POINTER(ctypes.c_uint8))

        # Build CUDA sync structs (fields default to 0)
        access_sync = bindings.ovrtx_cuda_sync_t()
        done_sync = bindings.ovrtx_cuda_sync_t()
        if cuda_stream is not None:
            access_sync.stream = cuda_stream
            done_sync.stream = cuda_stream
        if cuda_event is not None:
            access_sync.wait_event = cuda_event

        # Build input_buffer pointing to array
        self.input_buffer = bindings.ovrtx_input_buffer_t(
            tensors=self._tensor_array,  # Array reference kept in self
            tensor_count=len(self._tensor_storage),
            dirty_bits=dirty_bits_ptr,
            dirty_bits_size=dirty_bits_size,
            access_cuda_sync=access_sync,
            done_cuda_sync=done_sync,
        )


class Renderer:
    """High-level Pythonic renderer for OVRTX.

    Wraps the C library with automatic resource management. Resources are
    cleaned up automatically when the renderer goes out of scope.

    Enum types for method parameters are accessible as class attributes
    (e.g., ``Renderer.PrimMode.EXISTING_ONLY``) or via top-level imports
    (e.g., ``from ovrtx import PrimMode``).

    Example:
        ```python
        from ovrtx import Renderer, RendererConfig

        # Use defaults
        renderer = Renderer()

        # Or customize
        config = RendererConfig(sync_mode=True, log_file_path="/path/to/app.log")
        renderer = Renderer(config=config)

        # Resources automatically cleaned up when renderer goes out of scope
        ```
    """

    Semantic = Semantic
    PrimMode = PrimMode
    DataAccess = DataAccess
    Device = Device

    def __init__(self, config: Optional[RendererConfig] = None):
        """Create a new renderer instance.

        Args:
            config: Optional renderer configuration. If None, uses default configuration.

        Raises:
            RuntimeError: If the renderer cannot be created.
        """
        self._handle: Any = None

        # Use default config if none provided, otherwise copy to avoid mutating user's object
        if config is None:
            config = RendererConfig()
        else:
            from copy import copy

            config = copy(config)

        # get bindings to ovrtx library (including lazy-loading and initializing the library if necessary)
        ovrtx_config = self._to_c_config(config)
        _bindings = bindings._ovrtx_loader.create_bindings(ovrtx_config)

        # Create a new renderer instance
        result, c_renderer = _bindings.create_renderer(ovrtx_config)
        if result.status != bindings.OVRTX_API_SUCCESS:
            error_msg = _bindings.get_last_error() or "Unknown error"
            raise RuntimeError(f"Failed to create renderer: {error_msg}\nconfig: {config}")

        # Store bindings, renderer handle and config for other methods to use
        self._handle = c_renderer
        self._bindings = _bindings
        self._config = config

    def __del__(self):
        """Automatically clean up renderer resources on destruction."""
        if self._handle is not None:
            try:
                result = self._bindings.destroy_renderer(self._handle)

                # Report failure (raise so it's caught and reported below)
                if result.status != bindings.OVRTX_API_SUCCESS:
                    error_msg = self._bindings.get_last_error() or "Unknown error"
                    raise RuntimeError(f"Failed to destroy renderer: {error_msg}")
            except Exception as e:
                # Report any exceptions to stderr (can't raise from __del__)
                print(f"Warning: Exception during renderer cleanup in __del__: {e}", file=sys.stderr)
            finally:
                # Always clear references to prevent double-cleanup attempts
                self._handle = None
                self._bindings = None
                self._config = None

    @property
    def version(self) -> tuple:
        """Runtime library version as a (major, minor, patch) tuple."""
        return self._bindings.version

    def add_usd(self, usd_file_path: str, path_prefix: Optional[str] = None) -> Any:
        """Add a USD file to the renderer (synchronous).

        Blocks until the USD file is fully loaded.

        Args:
            usd_file_path: Path to the USD file to add.
            path_prefix: Optional path prefix for the USD stage.

        Returns:
            USD handle for the loaded stage.

        Raises:
            RuntimeError: If renderer is invalid, enqueue fails, or loading fails.

        Example:
            ```python
            usd_handle = renderer.add_usd("/path/to/scene.usda")
            ```
        """
        return self.add_usd_async(usd_file_path, path_prefix).wait()

    def add_usd_async(self, usd_file_path: str, path_prefix: Optional[str] = None) -> Operation[Any]:
        """Add a USD file to the renderer (asynchronous).

        Returns immediately with an Operation for manual control.
        Advanced users can use this for custom timeout handling or polling.

        Args:
            usd_file_path: Path to the USD file to add.
            path_prefix: Optional path prefix for the USD stage.

        Returns:
            Operation that can be polled or waited on with custom timeout.

        Raises:
            RuntimeError: If renderer is invalid or enqueue fails.

        Example:
            ```python
            # Poll with custom timeout
            op = renderer.add_usd_async("/path/to/scene.usda")
            usd_handle = op.wait(timeout_ns=1_000_000_000)  # 1 second
            if usd_handle is None:
                print("Still loading...")
            else:
                print(f"Loaded: {usd_handle}")
            ```
        """
        if self._handle is None:
            raise RuntimeError("Renderer is not valid")

        # Create USD input descriptor for file path mode
        usd_input = bindings.ovrtx_usd_input_t(
            usd_file_path=bindings.ovx_string_t(usd_file_path),
            usd_stage_id=0,
            usd_layer_content=bindings.ovx_string_t(),
        )

        result, usd_handle = self._bindings.add_usd(self._handle, usd_input, path_prefix)
        if result.status != bindings.OVRTX_API_SUCCESS:
            error_msg = self._bindings.get_last_error() or "Unknown error"
            raise RuntimeError(f"Failed to enqueue USD load: {error_msg}")

        return Operation(
            renderer=self, op_id=result.op_index.value, handle=usd_handle, operation_name=f"add_usd({usd_file_path})"
        )

    def add_usd_layer(self, usd_layer_content: str, path_prefix: Optional[str] = None) -> Any:
        """Add inline USD layer content to the renderer (synchronous).

        Blocks until the USD layer is fully loaded.

        Args:
            usd_layer_content: USDA content as a Python string. Must set defaultPrim
                for reference composition to work correctly.
            path_prefix: Path where layer is composed. Absolute paths in layer content
                must match this prefix. Cannot reuse an existing prim path.

        Returns:
            USD handle for the loaded layer.

        Raises:
            RuntimeError: If renderer is invalid, enqueue fails, or loading fails.
            ValueError: If usd_layer_content is empty.

        Example:
            ```python
            renderer.add_usd("/path/to/scene.usda")
            renderer.add_usd_layer('''
            #usda 1.0
            (defaultPrim = "Render")
            def Scope "Render" {
                def RenderProduct "Camera" { rel camera = </World/Camera> }
            }
            ''', path_prefix="/Render")
            ```
        """
        return self.add_usd_layer_async(usd_layer_content, path_prefix).wait()

    def add_usd_layer_async(self, usd_layer_content: str, path_prefix: Optional[str] = None) -> Operation[Any]:
        """Add inline USD layer content to the renderer (asynchronous).

        Returns immediately with an Operation for manual control.

        Args:
            usd_layer_content: USDA content as a Python string. Must set defaultPrim
                for reference composition to work correctly.
            path_prefix: Path where layer is composed. Absolute paths in layer content
                must match this prefix. Cannot reuse an existing prim path.

        Returns:
            Operation that can be polled or waited on with custom timeout.

        Raises:
            RuntimeError: If renderer is invalid or enqueue fails.
            ValueError: If usd_layer_content is empty.

        Example:
            ```python
            op = renderer.add_usd_layer_async('''
            #usda 1.0
            (defaultPrim = "Inline")
            def Xform "Inline" { ... }
            ''', path_prefix="/Inline")
            op.wait()
            ```
        """
        if self._handle is None:
            raise RuntimeError("Renderer is not valid")

        if not usd_layer_content or not usd_layer_content.strip():
            raise ValueError("usd_layer_content cannot be empty")

        # Create USD input descriptor for layer content mode
        # Note: usd_file_path is empty, usd_layer_content is populated
        usd_input = bindings.ovrtx_usd_input_t(
            usd_file_path=bindings.ovx_string_t(),  # Empty - not using file path
            usd_stage_id=0,
            usd_layer_content=bindings.ovx_string_t(usd_layer_content),
        )

        result, usd_handle = self._bindings.add_usd(self._handle, usd_input, path_prefix)
        if result.status != bindings.OVRTX_API_SUCCESS:
            error_msg = self._bindings.get_last_error() or "Unknown error"
            raise RuntimeError(f"Failed to enqueue USD layer load: {error_msg}")

        return Operation(
            renderer=self, op_id=result.op_index.value, handle=usd_handle, operation_name="add_usd_layer(<inline>)"
        )

    def remove_usd(self, usd_handle: Any) -> None:
        """Remove a previously added USD from the runtime stage.

        Args:
            usd_handle: Handle returned from add_usd() or add_usd_layer().

        Raises:
            RuntimeError: If the removal fails.
        """
        self.remove_usd_async(usd_handle).wait()

    def remove_usd_async(self, usd_handle: Any) -> Operation[None]:
        """Remove a previously added USD (async).

        Args:
            usd_handle: Handle returned from add_usd() or add_usd_layer().

        Returns:
            Operation that completes when USD is removed.

        Raises:
            RuntimeError: If renderer is invalid or enqueue fails.
        """
        if self._handle is None:
            raise RuntimeError("Renderer is not valid")

        result = self._bindings.remove_usd(self._handle, usd_handle)
        if result.status != bindings.OVRTX_API_SUCCESS:
            error_msg = self._bindings.get_last_error() or "Unknown error"
            raise RuntimeError(f"Failed to enqueue remove_usd: {error_msg}")

        return Operation(renderer=self, op_id=result.op_index.value, handle=None, operation_name="remove_usd")

    def update_from_usd_time(self, usd_time: float) -> None:
        """Update runtime stage to a specific USD time.

        Updates all time-sampled attributes in the runtime stage to the provided USD time.
        Use this for animated USD scenes to evaluate attributes at different times.

        Args:
            usd_time: USD time to update the stage to.

        Raises:
            RuntimeError: If the update fails.
        """
        if self._handle is None:
            raise RuntimeError("Renderer is not valid")

        result = self._bindings.update_stage_from_usd_time(self._handle, usd_time)
        if result.status != bindings.OVRTX_API_SUCCESS:
            error_msg = self._bindings.get_last_error() or "Unknown error"
            raise RuntimeError(f"Failed to enqueue update_from_usd_time: {error_msg}")

        op = Operation(renderer=self, op_id=result.op_index.value, handle=None, operation_name="update_from_usd_time")
        op.wait()

    def clone_usd(self, source_path: str, target_paths: List[str]) -> None:
        """Clone a USD subtree to one or more target paths.

        Creates copies of the prim at source_path at each target path.

        Args:
            source_path: Path to the source prim to clone.
            target_paths: List of target paths for the clones.

        Raises:
            RuntimeError: If the clone operation fails.
            ValueError: If target_paths is empty.
        """
        self.clone_usd_async(source_path, target_paths).wait()

    def clone_usd_async(self, source_path: str, target_paths: List[str]) -> Operation[None]:
        """Clone a USD subtree (async).

        Args:
            source_path: Path to the source prim to clone.
            target_paths: List of target paths for the clones.

        Returns:
            Operation that completes when cloning is done.

        Raises:
            RuntimeError: If renderer is invalid or enqueue fails.
            ValueError: If target_paths is empty.
        """
        if self._handle is None:
            raise RuntimeError("Renderer is not valid")

        if not target_paths:
            raise ValueError("At least one target path is required")

        source_string = bindings.ovx_string_t(source_path)
        target_strings = [bindings.ovx_string_t(path) for path in target_paths]
        target_array = (bindings.ovx_string_t * len(target_strings))(*target_strings)

        result = self._bindings.clone_usd(self._handle, source_string, target_array, len(target_strings))
        if result.status != bindings.OVRTX_API_SUCCESS:
            error_msg = self._bindings.get_last_error() or "Unknown error"
            raise RuntimeError(f"Failed to enqueue clone_usd: {error_msg}")

        op = Operation(renderer=self, op_id=result.op_index.value, handle=None, operation_name="clone_usd")
        op._storage_refs = [source_string, target_strings, target_array]
        return op

    def _wait_operation(self, operation: Operation, timeout_ns: Optional[int]) -> Operation._Result:
        """Internal: Wait for an operation to complete.

        Args:
            operation: The operation to wait for.
            timeout_ns: Timeout in nanoseconds. None or negative means infinite timeout.

        Returns:
            Operation._Result with status and any errors.
        """
        # Use infinite timeout if None or negative
        if timeout_ns is None or timeout_ns < 0:
            timeout = bindings.OVRTX_TIMEOUT_INFINITE
        else:
            timeout = bindings.ovrtx_timeout_t(time_out_ns=timeout_ns)

        result, c_wait_result = self._bindings.wait_op(self._handle, bindings.ovrtx_op_id_t(operation.op_id), timeout)

        # Determine result state
        errors = []
        has_errors = False
        is_timeout = False

        if result.status != bindings.OVRTX_API_SUCCESS:
            if result.status == bindings.OVRTX_API_TIMEOUT:
                is_timeout = True
            else:
                has_errors = True
                error_msg = self._bindings.get_last_error() or "Unknown error"
                errors.append(error_msg)

        # Extract per-operation errors from wait_result
        if c_wait_result.num_error_ops > 0:
            has_errors = True
            for i in range(c_wait_result.num_error_ops):
                failed_op_id = c_wait_result.error_op_ids[i]
                error_msg = self._bindings.get_last_op_error(failed_op_id)
                errors.append(f"op {failed_op_id.value}: {error_msg}")

        # Set value and errors based on outcome
        # Status is inferred: errors → failed, value=None → timeout, value → completed
        return Operation._Result(value=None if (has_errors or is_timeout) else operation._handle, errors=errors)

    def step(self, render_products: set[str], delta_time: float) -> RenderProductSetOutputs[Any]:
        """Step the renderer (synchronous - blocks until complete).

        Enqueues a step operation, waits for rendering to complete, and returns results.
        This is the simple, blocking interface suitable for most use cases.

        Args:
            render_products: Set of render product paths to step.
            delta_time: Time delta for the simulation step.

        Returns:
            RenderProductSetOutputs with rendering results.

        Raises:
            RuntimeError: If renderer is invalid, enqueue fails, or rendering fails.
            ValueError: If no valid render products provided.

        Example:
            ```python
            products = renderer.step(render_products={"/Render/..."}, delta_time=0.1)
            for product_name, product in products.products.items():
                print(f"Rendered {product_name}")
            ```
        """
        return self.step_async(render_products, delta_time).wait()

    def step_async(self, render_products: set[str], delta_time: float) -> RendererResult[Any]:
        """Step the renderer (asynchronous - returns immediately).

        Enqueues a step operation and returns result for advanced control.
        Use this for custom timeout handling or polling.

        Args:
            render_products: Set of render product paths to step.
            delta_time: Time delta for the simulation step.

        Returns:
            RendererResult that can be waited on with custom timeouts.

        Raises:
            RuntimeError: If renderer is invalid or enqueue fails.
            ValueError: If no valid render products provided.

        Example:
            ```python
            result = renderer.step_async(render_products={"/Render/..."}, delta_time=0.1)
            # Poll without blocking
            metadata = result.wait(step_timeout_ns=0)
            if metadata is None:
                print("Still rendering...")
            else:
                print("Done!")
            ```
        """
        if self._handle is None:
            raise RuntimeError("Renderer is not valid")

        # Convert and filter render products
        render_products_strings = [
            bindings.ovx_string_t(prod) for prod in render_products if prod and str(prod).strip()
        ]

        if not render_products_strings:
            raise ValueError("At least one valid render product is required to step the renderer")

        # Create render product set from intermediate C array
        render_products_array = (bindings.ovx_string_t * len(render_products_strings))(*render_products_strings)
        render_product_set = bindings.ovrtx_render_product_set_t(
            render_products=render_products_array, num_render_products=len(render_products_strings)
        )

        # Enqueue step operation
        result, step_result_handle = self._bindings.step(self._handle, render_product_set, delta_time)

        if result.status != bindings.OVRTX_API_SUCCESS:
            error_msg = self._bindings.get_last_error() or "Unknown error"
            raise RuntimeError(f"Failed to enqueue step: {error_msg}")

        operation = Operation(
            renderer=self,
            op_id=result.op_index.value,
            handle=step_result_handle,
            operation_name=f"step(dt={delta_time})",
        )

        return RendererResult(operation)

    def reset(self, time: float = 0.0) -> None:
        """Reset sensor simulation history to a specific time.

        Clears accumulated rendering history for all render products and
        sets the simulation start time for future step() calls.

        Args:
            time: Simulation time to reset to (default: 0.0).

        Raises:
            RuntimeError: If the reset fails.
        """
        self.reset_async(time).wait()

    def reset_async(self, time: float = 0.0) -> Operation[None]:
        """Reset sensor simulation history (async).

        Args:
            time: Simulation time to reset to (default: 0.0).

        Returns:
            Operation that completes when reset is done.

        Raises:
            RuntimeError: If renderer is invalid or enqueue fails.
        """
        if self._handle is None:
            raise RuntimeError("Renderer is not valid")

        result = self._bindings.reset(self._handle, time)
        if result.status != bindings.OVRTX_API_SUCCESS:
            error_msg = self._bindings.get_last_error() or "Unknown error"
            raise RuntimeError(f"Failed to enqueue reset: {error_msg}")

        return Operation(renderer=self, op_id=result.op_index.value, handle=None, operation_name="reset")

    def reset_stage(self) -> None:
        """Reset stage to empty state.

        Clears all USD content from the runtime stage. After this call,
        the stage will be empty and new USD content can be loaded.

        Raises:
            RuntimeError: If the reset fails.
        """
        self.reset_stage_async().wait()

    def reset_stage_async(self) -> Operation[None]:
        """Reset stage to empty state (async).

        Returns:
            Operation that completes when stage is cleared.

        Raises:
            RuntimeError: If renderer is invalid or enqueue fails.
        """
        if self._handle is None:
            raise RuntimeError("Renderer is not valid")

        result = self._bindings.reset_stage(self._handle)
        if result.status != bindings.OVRTX_API_SUCCESS:
            error_msg = self._bindings.get_last_error() or "Unknown error"
            raise RuntimeError(f"Failed to enqueue reset_stage: {error_msg}")

        return Operation(renderer=self, op_id=result.op_index.value, handle=None, operation_name="reset_stage")

    def _fetch_results(
        self, operation: Operation[bindings.ovrtx_step_result_handle_t], timeout_ns: Optional[int]
    ) -> RenderProductSetOutputs[bindings.ovrtx_step_result_handle_t]:
        """Internal: Fetch rendering results metadata from C API.

        Args:
            operation: Step operation (handle is ovrtx_step_result_handle_t)
            timeout_ns: Timeout in nanoseconds (usually None after operation completes)

        Returns:
            RenderProductSetOutputs with parsed metadata

        Raises:
            RuntimeError: If fetch fails
        """
        if self._handle is None:
            raise RuntimeError("Renderer is not valid")

        # Prepare timeout
        if timeout_ns is None or timeout_ns < 0:
            timeout = bindings.OVRTX_TIMEOUT_INFINITE
        else:
            timeout = bindings.ovrtx_timeout_t(time_out_ns=timeout_ns)

        # Call C API and raise on error
        result, c_outputs = self._bindings.fetch_results(self._handle, operation.handle, timeout)
        if result.status != bindings.OVRTX_API_SUCCESS:
            error_msg = self._bindings.get_last_error() or "Unknown error"
            raise RuntimeError(f"Failed to fetch results: {error_msg}")

        # Parse C structures into Python dataclasses, filtering out unmappable subframe AOVs
        ignored_render_vars = ["LdrColor0", "LdrColor1", "LdrColor2", "HdrColor0", "HdrColor1", "HdrColor2"]
        products = {}
        for c_product in c_outputs.outputs[: c_outputs.output_count]:
            # Parse frames
            frames = []
            for c_frame in c_product.output_frames[: c_product.output_frame_count]:
                # Parse render vars
                render_vars = []
                for c_var in c_frame.output_render_vars[: c_frame.render_var_count]:
                    if (var_name := str(c_var.render_var_name)) not in ignored_render_vars:
                        var = RenderVarOutput(
                            name=var_name,
                            handle=c_var.output_handle,
                            renderer=self,
                        )
                        render_vars.append(var)

                frames.append(
                    FrameOutput(
                        start_time=c_frame.frame_start_time,
                        end_time=c_frame.frame_end_time,
                        render_vars={var.name: var for var in render_vars},
                    )
                )

            product_name = str(c_product.render_product_path)
            products[product_name] = ProductOutput(
                name=product_name,
                frames=frames,
            )

        return RenderProductSetOutputs(operation=operation, products=products)

    def _destroy_results(self, operation: Operation[bindings.ovrtx_step_result_handle_t]) -> None:
        """Internal: Destroy step results and free resources.

        Args:
            operation: Step operation (provides step result handle for cleanup)

        Raises:
            RuntimeError: If destroy fails
        """
        if self._handle is None:
            raise RuntimeError("Renderer is not valid")

        result = self._bindings.destroy_results(self._handle, operation.handle)

        if result.status != bindings.OVRTX_API_SUCCESS:
            error_msg = self._bindings.get_last_error() or "Unknown error"
            raise RuntimeError(f"Failed to destroy results: {error_msg}")

    def _map_output(
        self,
        output_handle: bindings.ovrtx_rendered_output_handle_t,
        device_type: int,
        sync_stream: Optional[int] = None,
    ) -> tuple[bindings.DLTensor, bindings.ovrtx_rendered_output_map_handle_t]:
        """Internal: Map rendered output to target memory domain.

        Args:
            output_handle: Rendered output handle
            device_type: Device type constant (OVRTX_MAP_DEVICE_TYPE_*).
            sync_stream: CUDA stream handle for render completion sync. The stream will wait
                for render to complete before any subsequent work executes. Default: 1 for
                CUDA (default stream), 0 for CPU.

        Returns:
            Tuple of (DLTensor with mapped pixel data, map_handle for unmapping)

        Raises:
            RuntimeError: If mapping fails
        """
        if self._handle is None:
            raise RuntimeError("Renderer is not valid")

        # Default sync_stream: 1 for CUDA types (default stream), 0 for CPU
        if sync_stream is None:
            sync_stream = 0 if device_type == bindings.OVRTX_MAP_DEVICE_TYPE_CPU else 1
        map_desc = bindings.ovrtx_map_output_description_t(device_type=device_type, sync_stream=sync_stream)
        result, rendered_output = self._bindings.map_rendered_output(
            self._handle, output_handle, ctypes.byref(map_desc), bindings.OVRTX_TIMEOUT_INFINITE
        )

        if result.status != bindings.OVRTX_API_SUCCESS:
            error_msg = self._bindings.get_last_error() or "Unknown error"
            raise RuntimeError(f"Failed to map output: {error_msg}")

        # Check if the rendered output itself is valid (not just the API call)
        if rendered_output.status != bindings.OVRTX_EVENT_COMPLETED:
            # Only extract error message if status indicates failure
            if rendered_output.status == bindings.OVRTX_EVENT_FAILURE:
                error_msg = str(rendered_output.error_message) if rendered_output.error_message.ptr else "Unknown error"
                raise RuntimeError(f"Rendered output failed: {error_msg}")
            else:
                # PENDING or other non-completed status
                raise RuntimeError(f"Rendered output not ready (status={rendered_output.status})")

        # Status is COMPLETED - buffer is valid, don't touch error_message field
        return rendered_output.buffer.dl, rendered_output.map_handle

    def _unmap_output(
        self,
        map_handle: bindings.ovrtx_rendered_output_map_handle_t,
        before_destroy_cuda_sync: Optional[bindings.ovrtx_cuda_sync_t] = None,
    ) -> None:
        """Internal: Unmap rendered output.

        Args:
            map_handle: Map handle from map operation
            before_destroy_cuda_sync: Optional CUDA sync to wait for before destroying/reusing
                the mapped memory. Pass None for no sync (default).

        Raises:
            RuntimeError: If unmapping fails
        """
        if self._handle is None:
            raise RuntimeError("Renderer is not valid")

        result = self._bindings.unmap_rendered_output(self._handle, map_handle, before_destroy_cuda_sync)

        if result.status != bindings.OVRTX_API_SUCCESS:
            error_msg = self._bindings.get_last_error() or "Unknown error"
            raise RuntimeError(f"Failed to unmap output: {error_msg}")

    @staticmethod
    def _to_c_config(config: RendererConfig) -> bindings.ovrtx_config_t:
        """Convert RendererConfig to C structure.

        Uses a whitelist of known config fields. Only whitelisted fields are converted
        to C config entries.
        """
        # Whitelist: field_name -> (factory_function, config_key_enum)
        # Field names must match RendererConfig dataclass fields.
        # Config key enums must match their C header counterparts in both name and
        # integer value (see ovrtx_config_bool_t / ovrtx_config_string_t in ovrtx_types.h).
        # Factories accept native Python types (bool / str) and handle lifetime internally.
        WHITELIST = {
            "sync_mode": (bindings.ovrtx_config_entry_bool, bindings.OVRTX_CONFIG_SYNC_MODE),
            "log_file_path": (bindings.ovrtx_config_entry_string, bindings.OVRTX_CONFIG_LOG_FILE_PATH),
            "log_level": (bindings.ovrtx_config_entry_string, bindings.OVRTX_CONFIG_LOG_LEVEL),
            "enable_profiling": (bindings.ovrtx_config_entry_bool, bindings.OVRTX_CONFIG_ENABLE_PROFILING),
            "read_gpu_transforms": (bindings.ovrtx_config_entry_bool, bindings.OVRTX_CONFIG_READ_GPU_TRANSFORMS),
            "output_partial_frames": (bindings.ovrtx_config_entry_bool, bindings.OVRTX_CONFIG_OUTPUT_PARTIAL_FRAMES),
            "keep_system_alive": (bindings.ovrtx_config_entry_bool, bindings.OVRTX_CONFIG_KEEP_SYSTEM_ALIVE),
            "active_cuda_gpus": (bindings.ovrtx_config_entry_string, bindings.OVRTX_CONFIG_ACTIVE_CUDA_GPUS),
        }

        entries: List[bindings.ovrtx_config_entry_t] = []
        for field_name, (factory, key) in WHITELIST.items():
            value = getattr(config, field_name, None)
            if value is not None:
                entries.append(factory(key, value))

        return bindings.ovrtx_config_t(entries)

    @staticmethod
    def _normalize_tensor_for_write(semantic_constant: int, tensor: DLTensor) -> DLTensor:
        """Validate and optionally pass through a DLTensor for write_attribute.

        For semantics that support layout-compatible input (e.g. XFORM_MAT4x4), tensors
        with a single component and the expected trailing shape (e.g. (N, 4, 4)) are
        passed through to the C API, which accepts them via the layout-compatible path.
        Tensors with the canonical multi-component dtype (e.g. 16 components) are also
        passed through unchanged. Strides are preserved so that a non-zero stride on
        the first dimension is supported by the C API.

        Args:
            semantic_constant: Parsed semantic constant (OVRTX_SEMANTIC_*).
            tensor: Input DLTensor to validate and optionally pass through.

        Returns:
            The same tensor (pass-through) when shape/dtype are valid for this semantic.

        Raises:
            ValueError: If tensor has one component but shape does not match the
                       expected layout for the given semantic.
        """
        # Layout-compatible rules: semantic → (source_dtype code/bits, trailing_shape_dims, target_lanes).
        _LAYOUT_RULES = {
            bindings.OVRTX_SEMANTIC_XFORM_MAT4x4: (
                DLDataType(code=DLDataTypeCode.kDLFloat, bits=64, lanes=1),
                (4, 4),
                16,
            ),
        }

        if (rule := _LAYOUT_RULES.get(semantic_constant)) is None:
            return tensor  # No rules for NONE or unknown semantics

        source_dtype, trailing_dims, target_lanes = rule
        expected_ndim = 1 + len(trailing_dims)

        # Pass-through: tensor already has the canonical multi-lane dtype (e.g. shape (N,) lanes=16).
        if tensor.dtype.lanes == target_lanes:
            return tensor

        # Pass-through: layout-compatible (lanes=1, shape (N, 4, 4)). C API accepts this directly.
        if tensor.dtype.lanes != 1:
            return tensor

        if tensor.dtype.code.value != source_dtype.code.value or tensor.dtype.bits != source_dtype.bits:
            return tensor  # Let _resolve_semantic_and_dtype catch the mismatch

        if tensor.shape is not None and tensor.ndim == expected_ndim:
            actual_trailing = tuple(tensor.shape[1 + i] for i in range(len(trailing_dims)))
            if actual_trailing == trailing_dims:
                return tensor  # Pass through to C layout-compatible path (keeps strides)

        _shape_str = "unknown" if tensor.shape is None else str(tuple(tensor.shape[i] for i in range(tensor.ndim)))
        _expected_shape = "(N, " + ", ".join(str(d) for d in trailing_dims) + ")"
        raise ValueError(
            f"write_attribute: tensor shape {_shape_str} does not match expected shape "
            f"{_expected_shape} for this semantic. Provide a tensor with the correct shape."
        )

    @staticmethod
    def _is_string_semantic(semantic: int) -> bool:
        """Return True if the semantic represents a string type (token or path)."""
        return semantic in (Semantic.TOKEN_STRING, Semantic.PATH_STRING)

    @staticmethod
    def _is_layout_compatible_semantic(semantic: int) -> bool:
        """Return True if the semantic allows layout-compatible tensor input (same code/bits, component count may differ)."""
        return semantic == Semantic.XFORM_MAT4x4

    @staticmethod
    def _strings_to_dltensor(strings: list[str]) -> DLTensor:
        """Convert a list of Python strings to a DLTensor of ovx_string_t structs.

        Each string is encoded as UTF-8 and stored in an ovx_string_t (ptr + length).
        The resulting DLTensor has dtype (kDLUInt, 128, 1) matching sizeof(ovx_string_t).

        Args:
            strings: List of Python strings to convert.

        Returns:
            DLTensor wrapping a contiguous array of ovx_string_t. Internal storage
            (byte buffers and shape array) is kept alive via _string_conversion_storage.
        """
        n = len(strings)
        string_array = (bindings.ovx_string_t * n)()
        for i, s in enumerate(strings):
            string_array[i] = bindings.ovx_string_t(s)

        shape_array = (ctypes.c_int64 * 1)(n)

        tensor = DLTensor(
            data=ctypes.cast(string_array, ctypes.c_void_p),
            device=DLDevice(device_type=DLDeviceType.kDLCPU, device_id=0),
            ndim=1,
            dtype=DLDataType(code=DLDataTypeCode.kDLUInt, bits=128, lanes=1),
            shape=ctypes.cast(shape_array, ctypes.POINTER(ctypes.c_int64)),
            strides=None,
            byte_offset=0,
        )
        # Keep ctypes arrays alive for the lifetime of the DLTensor
        tensor._string_conversion_storage = {"string_array": string_array, "shape_array": shape_array}

        return tensor

    def _resolve_semantic_and_dtype(
        self, semantic: Semantic, dtype: Optional[DLDataType], context: str = "operation"
    ) -> tuple[int, DLDataType]:
        """Resolve dtype from semantic, validating any user-provided dtype.

        Semantics imply specific dtype requirements determined by the C library's
        internal representation. When a semantic is provided, the dtype is inferred
        automatically; user-provided dtypes are validated against the expected type.

        Args:
            semantic: Semantic enum value.
            dtype: User-provided dtype (optional if semantic implies dtype).
            context: Context string for error messages.

        Returns:
            Tuple of (semantic_constant, semantic_dtype).

        Raises:
            ValueError: If dtype is required but not provided,
                       or if provided dtype doesn't match semantic.
        """
        if semantic == Semantic.XFORM_MAT4x4:
            inferred_dtype = DLDataType(code=DLDataTypeCode.kDLFloat, bits=64, lanes=16)
            layout_compatible = True  # C API accepts (code, bits) match; lanes may differ (e.g. 1 with shape (N,4,4))
        elif semantic in (Semantic.TOKEN_STRING, Semantic.PATH_STRING):
            inferred_dtype = DLDataType(code=DLDataTypeCode.kDLUInt, bits=128, lanes=1)
            layout_compatible = False
        else:
            inferred_dtype = None  # NONE — user must provide dtype
            layout_compatible = False

        if inferred_dtype is not None:
            if dtype is not None:
                code_bits_ok = dtype.code.value == inferred_dtype.code.value and dtype.bits == inferred_dtype.bits
                if layout_compatible:
                    if not code_bits_ok:
                        raise ValueError(
                            f"{context}: provided dtype (code={dtype.code}, bits={dtype.bits}, "
                            f"components={dtype.lanes}) must match scalar type for {semantic!r} "
                            f"(code={inferred_dtype.code}, bits={inferred_dtype.bits}); "
                            f"component count may differ for layout-compatible input "
                            f"(e.g. shape (N, 4, 4) with 1 component)."
                        )
                else:
                    if not (code_bits_ok and dtype.lanes == inferred_dtype.lanes):
                        raise ValueError(
                            f"{context}: provided dtype (code={dtype.code}, bits={dtype.bits}, "
                            f"components={dtype.lanes}) doesn't match required dtype for {semantic!r} "
                            f"(code={inferred_dtype.code}, bits={inferred_dtype.bits}, "
                            f"components={inferred_dtype.lanes})"
                        )
            semantic_dtype = inferred_dtype
        elif dtype is None:
            raise ValueError(f"{context}: dtype is required when semantic is Semantic.NONE")
        else:
            semantic_dtype = dtype

        return (semantic, semantic_dtype)

    _DTYPE_MAP = {
        "float16": (DLDataTypeCode.kDLFloat, 16),
        "float32": (DLDataTypeCode.kDLFloat, 32),
        "float64": (DLDataTypeCode.kDLFloat, 64),
        "int8": (DLDataTypeCode.kDLInt, 8),
        "int16": (DLDataTypeCode.kDLInt, 16),
        "int32": (DLDataTypeCode.kDLInt, 32),
        "int64": (DLDataTypeCode.kDLInt, 64),
        "uint8": (DLDataTypeCode.kDLUInt, 8),
        "uint16": (DLDataTypeCode.kDLUInt, 16),
        "uint32": (DLDataTypeCode.kDLUInt, 32),
        "uint64": (DLDataTypeCode.kDLUInt, 64),
        "bool": (DLDataTypeCode.kDLBool, 8),
    }

    _PYTHON_BUILTIN_DTYPE_MAP = {
        float: "float64",
        int: "int64",
        bool: "bool",
    }

    _KIND_MAP = {
        "f": DLDataTypeCode.kDLFloat,
        "i": DLDataTypeCode.kDLInt,
        "u": DLDataTypeCode.kDLUInt,
        "b": DLDataTypeCode.kDLBool,
    }

    @staticmethod
    def _resolve_new_dtype_and_shape(
        dtype: Any, shape: Optional[tuple], context: str = "operation"
    ) -> tuple[int, DLDataType, Optional[tuple]]:
        """Translate new-style dtype + shape to (semantic_constant, DLDataType, shape).

        Args:
            dtype: Element type. Accepts string names (``"float32"``, ``"token"``, ``"path"``),
                NumPy dtypes (``np.float32``, ``np.dtype("float64")``), any object with
                ``.kind``/``.itemsize`` attributes (CuPy, JAX dtypes), or Python built-ins
                (``float``, ``int``).
            shape: Optional element shape tuple (e.g. (3,) for float3, (4, 4) for matrix4d).
            context: Context string for error messages.

        Returns:
            Tuple of (semantic_constant, resolved_DLDataType, shape).
        """
        if isinstance(dtype, str):
            if dtype == "token":
                return (Semantic.TOKEN_STRING, DLDataType(code=DLDataTypeCode.kDLUInt, bits=128, lanes=1), None)
            if dtype == "path":
                return (Semantic.PATH_STRING, DLDataType(code=DLDataTypeCode.kDLUInt, bits=128, lanes=1), None)
            dtype_str = dtype
        elif dtype in Renderer._PYTHON_BUILTIN_DTYPE_MAP:
            dtype_str = Renderer._PYTHON_BUILTIN_DTYPE_MAP[dtype]
        elif isinstance(dtype, type) and Renderer._DTYPE_MAP.get(getattr(dtype, "__name__", "")) is not None:
            dtype_str = dtype.__name__
        else:
            # Duck-type: np.dtype instance (has .kind + .itemsize directly)
            if hasattr(dtype, "kind") and hasattr(dtype, "itemsize"):
                code = Renderer._KIND_MAP.get(dtype.kind)
                if code is None:
                    raise ValueError(
                        f"{context}: dtype={dtype!r} is not a supported numeric type. "
                        f"Use a float, int, uint, or bool dtype (e.g. np.float32, np.int32, np.uint8, np.bool_)."
                    )
                bits = dtype.itemsize * 8
                lanes = math.prod(shape) if shape else 1
                return (Semantic.NONE, DLDataType(code=code, bits=bits, lanes=lanes), shape if shape else None)
            # Duck-type: objects with a .dtype property (e.g. numpy scalar type instances)
            if hasattr(dtype, "dtype") and hasattr(dtype.dtype, "kind") and hasattr(dtype.dtype, "itemsize"):
                code = Renderer._KIND_MAP.get(dtype.dtype.kind)
                if code is None:
                    raise ValueError(
                        f"{context}: dtype={dtype!r} is not a supported numeric type. "
                        f"Use a float, int, uint, or bool dtype (e.g. np.float32, np.int32, np.uint8, np.bool_)."
                    )
                bits = dtype.dtype.itemsize * 8
                lanes = math.prod(shape) if shape else 1
                return (Semantic.NONE, DLDataType(code=code, bits=bits, lanes=lanes), shape if shape else None)
            dtype_str = str(dtype)

        entry = Renderer._DTYPE_MAP.get(dtype_str)
        if entry is None:
            raise ValueError(
                f"{context}: unrecognized dtype={dtype!r}. Use a numpy dtype, Python built-in (float, int), "
                f"or a string name ('float32', 'int32', 'token', 'path', etc.)."
            )
        code, bits = entry
        lanes = math.prod(shape) if shape else 1
        return (Semantic.NONE, DLDataType(code=code, bits=bits, lanes=lanes), shape if shape else None)

    def _create_semantic_aware_dltensor(
        self, original_dl_tensor: DLTensor, semantic: int, element_shape: Optional[tuple] = None
    ) -> DLTensor:
        """Create reshaped DLTensor view for interop-friendly consumption.

        When ``element_shape`` is provided (new dtype/shape API path), reshapes
        ``(N,)`` with ``K`` components to ``(N, *element_shape)`` with a scalar element.
        When ``element_shape`` is None, applies semantic-specific reshaping
        (XFORM_MAT4x4 → ``(N, 4, 4)``).

        Args:
            original_dl_tensor: Original DLTensor from C API.
            semantic: Semantic constant (OVRTX_SEMANTIC_*).
            element_shape: Optional element shape from the new dtype/shape API.

        Returns:
            Reshaped DLTensor view (zero-copy metadata change), or the original
            tensor if no reshaping applies.
        """
        if original_dl_tensor.ndim < 1 or original_dl_tensor.shape is None:
            return original_dl_tensor

        N = original_dl_tensor.shape[0]
        lanes = original_dl_tensor.dtype.lanes

        # New path: reshape using stored element_shape
        if element_shape is not None and lanes > 1:
            new_dims = (N, *element_shape)
            ndim = len(new_dims)
            reshaped = DLTensor()
            reshaped.data = original_dl_tensor.data
            reshaped.device = original_dl_tensor.device
            reshaped.byte_offset = original_dl_tensor.byte_offset
            reshaped.strides = None
            reshaped.ndim = ndim
            shape_array = (ctypes.c_int64 * ndim)(*new_dims)
            reshaped.shape = ctypes.cast(shape_array, ctypes.POINTER(ctypes.c_int64))
            reshaped.dtype.code = original_dl_tensor.dtype.code
            reshaped.dtype.bits = original_dl_tensor.dtype.bits
            reshaped.dtype.lanes = 1
            reshaped._shape_storage = shape_array
            return reshaped

        # Semantic-specific path: XFORM_MAT4x4 reshape (N,) lanes=16 → (N, 4, 4) lanes=1
        if semantic == bindings.OVRTX_SEMANTIC_XFORM_MAT4x4 and lanes == 16:
            reshaped = DLTensor()
            reshaped.data = original_dl_tensor.data
            reshaped.device = original_dl_tensor.device
            reshaped.byte_offset = original_dl_tensor.byte_offset
            reshaped.strides = None
            reshaped.ndim = 3
            shape_array = (ctypes.c_int64 * 3)(N, 4, 4)
            reshaped.shape = ctypes.cast(shape_array, ctypes.POINTER(ctypes.c_int64))
            reshaped.dtype.code = original_dl_tensor.dtype.code
            reshaped.dtype.bits = original_dl_tensor.dtype.bits
            reshaped.dtype.lanes = 1
            reshaped._shape_storage = shape_array
            return reshaped

        return original_dl_tensor

    def write_attribute(
        self,
        prim_paths: List[str],
        attribute_name: str,
        tensor: Any,
        semantic: Semantic = Semantic.NONE,
        dirty_bits: Optional[bytes] = None,
        prim_mode: PrimMode = PrimMode.EXISTING_ONLY,
        data_access: DataAccess = DataAccess.SYNC,
        cuda_stream: Optional[int] = None,
        cuda_event: Optional[int] = None,
    ) -> None:
        """Write scalar attribute data synchronously.

        Pass data directly — the element type and component count are inferred
        automatically from the input:

        - **Tensor input** (NumPy, Warp, or any ``__dlpack__``-compatible
          object): the element type is derived from the array's dtype and
          shape. For example, a ``float32`` array with shape ``(N, 3)``
          is interpreted as a 3-component float attribute.
        - **String input** (``list[str]``): automatically written as a token
          string attribute. One string per prim.

        For repeated writes to the same attribute, consider using
        ``bind_attribute()`` followed by ``binding.write()`` for better
        performance.

        Args:
            prim_paths: List of prim paths to write to.
            attribute_name: Name of the attribute.
            tensor: A NumPy array, Warp array, or any ``__dlpack__``-compatible
                object. One element per prim.
                For token strings, pass a ``list[str]`` instead.
            semantic: Optional attribute semantic (e.g.,
                ``Semantic.XFORM_MAT4x4``). When omitted (default), the
                element type is inferred from the input data.
            dirty_bits: Optional dirty bit array for selective updates.
            prim_mode: Prim binding mode (e.g., ``PrimMode.EXISTING_ONLY``).
            data_access: Data access mode. ``DataAccess.SYNC`` (default) copies
                input data immediately so the caller's buffer can be reused
                after this call returns. ``DataAccess.ASYNC`` references the
                caller's buffer until the operation completes (zero-copy). Not
                allowed with string data.
            cuda_stream: CUDA stream handle (``int``) for GPU synchronization.
            cuda_event: CUDA event handle (``int``) for GPU synchronization.

        Raises:
            RuntimeError: If renderer is invalid or write fails.
            ValueError: If invalid parameters.
            TypeError: If tensor type is not supported, or if string data is
                not a ``list[str]``.

        Example:
            ```python
            import numpy as np

            # Tensor writes — just pass the array
            points = np.random.randn(N, 3).astype(np.float32)
            renderer.write_attribute(prim_paths, "points", points)

            matrices = np.tile(np.eye(4, dtype=np.float64), (N, 1, 1))
            renderer.write_attribute(prim_paths, "xformOp:transform", matrices)

            # Token strings — auto-detected from list[str]
            renderer.write_attribute(prim_paths, "displayName", ["Cube", "Sphere"])
            ```
        """
        self.write_attribute_async(
            prim_paths=prim_paths,
            attribute_name=attribute_name,
            tensor=tensor,
            semantic=semantic,
            dirty_bits=dirty_bits,
            prim_mode=prim_mode,
            data_access=data_access,
            cuda_stream=cuda_stream,
            cuda_event=cuda_event,
        ).wait()

    def write_array_attribute(
        self,
        prim_paths: List[str],
        attribute_name: str,
        tensors: List[Any],
        semantic: Semantic = Semantic.NONE,
        is_token: bool = False,
        dirty_bits: Optional[bytes] = None,
        prim_mode: PrimMode = PrimMode.EXISTING_ONLY,
        data_access: DataAccess = DataAccess.SYNC,
        cuda_stream: Optional[int] = None,
        cuda_event: Optional[int] = None,
    ) -> None:
        """Write array attribute data synchronously.

        Pass data directly — the element type and component count are inferred
        automatically from the input:

        - **Tensor input**: Each tensor corresponds to one prim. The element
          type is derived from the array's dtype and shape. Lengths may differ
          per prim.
        - **String input** (``list[list[str]]``): defaults to path/relationship
          arrays. Set ``is_token=True`` for token arrays (e.g.,
          ``xformOpOrder``, ``apiSchemas``).

        Args:
            prim_paths: List of prim paths to write to.
            attribute_name: Name of the array attribute.
            tensors: List of tensors (NumPy arrays, Warp arrays, or any
                ``__dlpack__``-compatible object), one per prim. Lengths may
                differ but element dtype must match the USD attribute schema.
                For string arrays, pass ``list[list[str]]`` instead.
            semantic: Optional attribute semantic (e.g.,
                ``Semantic.PATH_STRING``). When omitted (default), the
                element type is inferred from the input data.
            is_token: When passing ``list[list[str]]`` data, set ``True`` to write
                token arrays instead of path/relationship arrays. Ignored for
                tensor data. Default ``False``.
            dirty_bits: Optional dirty bit array for selective updates.
            prim_mode: Prim binding mode (e.g., ``PrimMode.EXISTING_ONLY``).
            data_access: Data access mode. ``DataAccess.SYNC`` (default) copies input
                data immediately. ``DataAccess.ASYNC`` references the caller's buffer
                until the operation completes (zero-copy). Not allowed with string data.
            cuda_stream: CUDA stream handle (``int``) for GPU synchronization.
            cuda_event: CUDA event handle (``int``) for GPU synchronization.

        Raises:
            RuntimeError: If renderer is invalid or write fails.
            ValueError: If invalid parameters or dtype mismatch.
            TypeError: If any tensor type is not supported, or if string data
                is not ``list[str]``.

        Note:
            Tensor dtype must exactly match the USD attribute's element type:

            - ``int[]`` → ``np.int32``
            - ``float3[]`` → ``np.float32`` with shape ``(N, 3)``
            - ``double3[]`` → ``np.float64`` with shape ``(N, 3)``

        Example:
            ```python
            import numpy as np

            # Tensor array write
            face_counts = np.array([4, 4, 4], dtype=np.int32)
            renderer.write_array_attribute(["/World/Mesh"], "faceVertexCounts", [face_counts])

            # Relationship arrays — default for list[list[str]]
            renderer.write_array_attribute(["/World/Mesh"], "material:binding",
                [["path1", "path2"]])

            # Token arrays — explicit is_token override
            renderer.write_array_attribute(["/World/Mesh"], "xformOpOrder",
                [["xformOp:translate", "xformOp:rotateXYZ"]], is_token=True)
            ```
        """
        self.write_array_attribute_async(
            prim_paths=prim_paths,
            attribute_name=attribute_name,
            tensors=tensors,
            semantic=semantic,
            is_token=is_token,
            dirty_bits=dirty_bits,
            prim_mode=prim_mode,
            data_access=data_access,
            cuda_stream=cuda_stream,
            cuda_event=cuda_event,
        ).wait()

    def _write_attribute_by_binding(
        self,
        binding: AttributeBinding,
        tensors: List[Any],
        dirty_bits: Optional[bytes] = None,
        data_access: DataAccess = DataAccess.SYNC,
        cuda_stream: Optional[int] = None,
        cuda_event: Optional[int] = None,
    ) -> None:
        """Write attribute data using a binding handle synchronously (internal)."""
        self._write_attribute_by_binding_async(
            binding=binding,
            tensors=tensors,
            dirty_bits=dirty_bits,
            data_access=data_access,
            cuda_stream=cuda_stream,
            cuda_event=cuda_event,
        ).wait()

    def write_attribute_async(
        self,
        prim_paths: List[str],
        attribute_name: str,
        tensor: Any,
        semantic: Semantic = Semantic.NONE,
        dirty_bits: Optional[bytes] = None,
        prim_mode: PrimMode = PrimMode.EXISTING_ONLY,
        data_access: DataAccess = DataAccess.SYNC,
        cuda_stream: Optional[int] = None,
        cuda_event: Optional[int] = None,
    ) -> Operation[None]:
        """Write scalar attribute data asynchronously.

        See :meth:`write_attribute` for full documentation and examples.

        Returns:
            Operation for async control (yields None on completion).
        """
        return self._write_attribute_internal(
            prim_paths=prim_paths,
            attribute_name=attribute_name,
            tensors=[tensor],
            semantic=semantic,
            dirty_bits=dirty_bits,
            prim_mode=prim_mode,
            is_array=False,
            data_access=data_access,
            cuda_stream=cuda_stream,
            cuda_event=cuda_event,
        )

    def write_array_attribute_async(
        self,
        prim_paths: List[str],
        attribute_name: str,
        tensors: List[Any],
        semantic: Semantic = Semantic.NONE,
        is_token: bool = False,
        dirty_bits: Optional[bytes] = None,
        prim_mode: PrimMode = PrimMode.EXISTING_ONLY,
        data_access: DataAccess = DataAccess.SYNC,
        cuda_stream: Optional[int] = None,
        cuda_event: Optional[int] = None,
    ) -> Operation[None]:
        """Write array attribute data asynchronously.

        See :meth:`write_array_attribute` for full documentation and examples.

        Returns:
            Operation for async control (yields None on completion).
        """
        return self._write_attribute_internal(
            prim_paths=prim_paths,
            attribute_name=attribute_name,
            tensors=tensors,
            semantic=semantic,
            is_token=is_token,
            dirty_bits=dirty_bits,
            prim_mode=prim_mode,
            is_array=True,
            data_access=data_access,
            cuda_stream=cuda_stream,
            cuda_event=cuda_event,
        )

    def _write_attribute_internal(
        self,
        prim_paths: List[str],
        attribute_name: str,
        tensors: List[Any],
        semantic: Semantic,
        dirty_bits: Optional[bytes],
        prim_mode: PrimMode,
        is_array: bool,
        is_token: bool = False,
        data_access: DataAccess = DataAccess.SYNC,
        cuda_stream: Optional[int] = None,
        cuda_event: Optional[int] = None,
    ) -> Operation[None]:
        """Internal: Write attribute data (shared implementation for scalar and array)."""
        if self._handle is None:
            raise RuntimeError("Renderer is not valid")

        if not tensors:
            raise ValueError("tensors list cannot be empty")

        if semantic != Semantic.NONE:
            # Explicit semantic path: resolve dtype from the semantic constant
            if semantic == Semantic.PATH_STRING and not is_array:
                raise ValueError(
                    "Semantic.PATH_STRING requires write_array_attribute (is_array=True). "
                    "Use write_array_attribute(semantic=Semantic.PATH_STRING, ...) instead of write_attribute."
                )

            if self._is_string_semantic(semantic) and data_access == DataAccess.ASYNC:
                raise ValueError("String attributes (token, path) require DataAccess.SYNC")

            if self._is_string_semantic(semantic):
                dl_tensors = []
                for i, t in enumerate(tensors):
                    if not isinstance(t, list) or not all(isinstance(s, str) for s in t):
                        raise TypeError(
                            f"tensors[{i}]: expected a list of strings for semantic {semantic!r}, "
                            f"got {type(t).__name__}"
                        )
                    dl_tensors.append(self._strings_to_dltensor(t))
                tensors = dl_tensors
            else:
                tensors = [
                    self._normalize_tensor_for_write(
                        semantic, t if isinstance(t, DLTensor) else DLTensor.from_dlpack(t)
                    )
                    for t in tensors
                ]

            tensor_dtype = tensors[0].dtype
            for i, t in enumerate(tensors[1:], start=1):
                if (
                    t.dtype.code.value != tensor_dtype.code.value
                    or t.dtype.bits != tensor_dtype.bits
                    or t.dtype.lanes != tensor_dtype.lanes
                ):
                    raise ValueError(
                        f"All tensors must have the same element type. "
                        f"tensors[0] has (code={tensor_dtype.code.value}, bits={tensor_dtype.bits}, "
                        f"components={tensor_dtype.lanes}), but tensors[{i}] has "
                        f"(code={t.dtype.code.value}, bits={t.dtype.bits}, components={t.dtype.lanes})"
                    )

            resolved_semantic, semantic_dtype = self._resolve_semantic_and_dtype(
                semantic, tensor_dtype, context="write_attribute"
            )
        else:
            # New path: auto-detect input type
            first = tensors[0]

            if isinstance(first, list) and all(isinstance(s, str) for s in first):
                # String input
                if data_access == DataAccess.ASYNC:
                    raise ValueError("String data requires DataAccess.SYNC")

                if not is_array:
                    resolved_semantic = Semantic.TOKEN_STRING
                else:
                    resolved_semantic = Semantic.TOKEN_STRING if is_token else Semantic.PATH_STRING

                dl_tensors = []
                for i, t in enumerate(tensors):
                    if not isinstance(t, list) or not all(isinstance(s, str) for s in t):
                        raise TypeError(
                            f"tensors[{i}]: expected a list of strings (all elements must be list[str]), "
                            f"got {type(t).__name__}"
                        )
                    dl_tensors.append(self._strings_to_dltensor(t))
                tensors = dl_tensors
                semantic_dtype = DLDataType(code=DLDataTypeCode.kDLUInt, bits=128, lanes=1)
            else:
                # Tensor / __dlpack__ input — infer component count from shape
                dl_tensors = [t if isinstance(t, DLTensor) else DLTensor.from_dlpack(t) for t in tensors]
                tensors = dl_tensors

                ref_tensor = tensors[0]
                ref_dtype = ref_tensor.dtype

                for i, t in enumerate(tensors[1:], start=1):
                    if (
                        t.dtype.code.value != ref_dtype.code.value
                        or t.dtype.bits != ref_dtype.bits
                        or t.dtype.lanes != ref_dtype.lanes
                    ):
                        raise ValueError(
                            f"All tensors must have the same element type. "
                            f"tensors[0] has (code={ref_dtype.code.value}, bits={ref_dtype.bits}, "
                            f"components={ref_dtype.lanes}), but tensors[{i}] has "
                            f"(code={t.dtype.code.value}, bits={t.dtype.bits}, components={t.dtype.lanes})"
                        )

                # Infer component count from trailing shape dimensions and tensor lanes.
                # NumPy exports lanes=1 with shape encoding (e.g. (N,3) float32 → lanes=1).
                # Warp exports structured types with lanes>1 (e.g. mat44d → lanes=16, shape=(N,)).
                # Multiplying accounts for both: 1*3=3 for NumPy, 16*1=16 for Warp mat44d.
                if ref_tensor.ndim >= 2 and ref_tensor.shape is not None:
                    trailing_dims = math.prod(ref_tensor.shape[i] for i in range(1, ref_tensor.ndim))
                else:
                    trailing_dims = 1

                semantic_dtype = DLDataType(
                    code=ref_dtype.code, bits=ref_dtype.bits, lanes=ref_dtype.lanes * trailing_dims
                )
                resolved_semantic = Semantic.NONE

        input_storage = _InputBufferStorage(
            tensors, dirty_bits=dirty_bits, cuda_stream=cuda_stream, cuda_event=cuda_event
        )

        binding_storage = _AttributeBindingDescStorage(
            prim_paths=prim_paths,
            attribute_name=attribute_name,
            dtype=semantic_dtype,
            semantic=resolved_semantic,
            prim_mode=prim_mode,
            is_array=is_array,
        )

        result = self._bindings.write_attribute(
            self._handle,
            binding_storage.binding_desc_or_handle,
            input_storage.input_buffer,
            bindings.ovrtx_data_access_t(data_access),
        )

        if result.status != bindings.OVRTX_API_SUCCESS:
            error_msg = self._bindings.get_last_error() or "Unknown error"
            raise RuntimeError(f"Failed to enqueue attribute write: {error_msg}")

        op = Operation(
            renderer=self, op_id=result.op_index.value, handle=None, operation_name=f"write_attribute({attribute_name})"
        )

        if data_access == DataAccess.ASYNC:
            op._storage_refs = [input_storage, binding_storage]

        return op

    def _write_attribute_by_binding_async(
        self,
        binding: AttributeBinding,
        tensors: List[Any],
        dirty_bits: Optional[bytes] = None,
        data_access: DataAccess = DataAccess.SYNC,
        cuda_stream: Optional[int] = None,
        cuda_event: Optional[int] = None,
    ) -> Operation[None]:
        """Write attribute data using a binding handle asynchronously (returns Operation).

        Args:
            binding: AttributeBinding from bind_attribute().
            tensors: List of tensors (NumPy arrays, Warp arrays, or any
                ``__dlpack__``-compatible objects).
            dirty_bits: Optional dirty bit array for selective updates.
            data_access: Data access mode (DataAccess.SYNC or DataAccess.ASYNC).
            cuda_stream: Optional CUDA stream handle for synchronization.
            cuda_event: Optional CUDA event handle for synchronization.

        Returns:
            Operation for async control (yields None on completion).

        Raises:
            RuntimeError: If renderer is invalid or write fails.
        """
        if self._handle is None:
            raise RuntimeError("Renderer is not valid")

        if not tensors:
            raise ValueError("tensors list cannot be empty")

        if self._is_string_semantic(binding.semantic) and data_access == DataAccess.ASYNC:
            raise ValueError("String attributes (token, path) require DataAccess.SYNC")

        if self._is_string_semantic(binding.semantic):
            dl_tensors = []
            for i, t in enumerate(tensors):
                if not isinstance(t, list) or not all(isinstance(s, str) for s in t):
                    raise TypeError(
                        f"tensors[{i}]: expected a list of strings for this string attribute binding, "
                        f"got {type(t).__name__}"
                    )
                dl_tensors.append(self._strings_to_dltensor(t))
            tensors = dl_tensors
        else:
            tensors = [
                self._normalize_tensor_for_write(
                    binding.semantic, t if isinstance(t, DLTensor) else DLTensor.from_dlpack(t)
                )
                for t in tensors
            ]

        first_dtype = tensors[0].dtype
        for i, t in enumerate(tensors[1:], start=1):
            if (
                t.dtype.code.value != first_dtype.code.value
                or t.dtype.bits != first_dtype.bits
                or t.dtype.lanes != first_dtype.lanes
            ):
                raise ValueError(
                    f"All tensors must have the same element type. "
                    f"tensors[0] has (code={first_dtype.code.value}, bits={first_dtype.bits}, "
                    f"components={first_dtype.lanes}), but tensors[{i}] has "
                    f"(code={t.dtype.code.value}, bits={t.dtype.bits}, components={t.dtype.lanes})"
                )

        expected_dtype = binding.dtype
        code_bits_ok = first_dtype.code.value == expected_dtype.code.value and first_dtype.bits == expected_dtype.bits
        lanes_ok = first_dtype.lanes == expected_dtype.lanes
        if self._is_layout_compatible_semantic(binding.semantic) or binding.shape is not None:
            if not code_bits_ok:
                raise ValueError(
                    f"Tensor element type (code={first_dtype.code.value}, bits={first_dtype.bits}, "
                    f"components={first_dtype.lanes}) must match binding's scalar type "
                    f"(code={expected_dtype.code.value}, bits={expected_dtype.bits}); "
                    f"component count may differ for layout-compatible input."
                )
        elif not (code_bits_ok and lanes_ok):
            raise ValueError(
                f"Tensor element type (code={first_dtype.code.value}, bits={first_dtype.bits}, "
                f"components={first_dtype.lanes}) doesn't match binding's expected type "
                f"(code={expected_dtype.code.value}, bits={expected_dtype.bits}, "
                f"components={expected_dtype.lanes})"
            )

        input_storage = _InputBufferStorage(
            tensors, dirty_bits=dirty_bits, cuda_stream=cuda_stream, cuda_event=cuda_event
        )

        binding_desc_or_handle = bindings.ovrtx_binding_desc_or_handle_t(
            binding_desc=bindings.ovrtx_binding_desc_t(),
            binding_handle=bindings.ovrtx_attribute_binding_handle_t(binding.handle),
        )

        result = self._bindings.write_attribute(
            self._handle,
            binding_desc_or_handle,
            input_storage.input_buffer,
            bindings.ovrtx_data_access_t(data_access),
        )

        if result.status != bindings.OVRTX_API_SUCCESS:
            error_msg = self._bindings.get_last_error() or "Unknown error"
            raise RuntimeError(f"Failed to enqueue attribute write: {error_msg}")

        op = Operation(
            renderer=self,
            op_id=result.op_index.value,
            handle=None,
            operation_name=f"write_attribute(handle={binding.handle})",
        )

        if data_access == DataAccess.ASYNC:
            op._storage_refs = [input_storage]

        return op

    def bind_attribute(
        self,
        prim_paths: List[str],
        attribute_name: str,
        dtype: Any = None,
        shape: Optional[tuple] = None,
        semantic: Semantic = Semantic.NONE,
        prim_mode: PrimMode = PrimMode.EXISTING_ONLY,
    ) -> "AttributeBinding[DLTensor]":
        """Create a persistent binding for scalar attribute writes.

        Creates a binding handle that can be reused for multiple write() calls,
        avoiding the overhead of recreating the binding descriptor each time.

        Specify the attribute type using ``dtype`` and optionally ``shape``:

        - **Tensor types**: ``dtype="float32", shape=(3,)`` for ``point3f``,
          ``dtype="float64", shape=(4, 4)`` for ``matrix4d``, etc.
        - **String types**: ``dtype="token"`` or ``dtype="path"``.
        - **Scalars**: ``dtype="int32"`` (no shape needed).

        Args:
            prim_paths: List of prim paths.
            attribute_name: Name of the attribute.
            dtype: Element type. Accepts string names (``"float32"``,
                ``"float64"``, ``"int32"``, ``"token"``, ``"path"``),
                NumPy dtypes (``np.float32``, ``np.dtype("float64")``),
                any dtype object with ``.kind``/``.itemsize`` attributes
                (CuPy, JAX), Python built-ins (``float``, ``int``), or a
                ``DLDataType`` object.
            shape: Element shape tuple (e.g. ``(3,)`` for 3-component vectors,
                ``(4, 4)`` for matrices). Mandatory for multi-component types
                when using numpy dtypes or string names. Omit for scalars.
                Ignored for string types and ``DLDataType``.
            semantic: Optional attribute semantic (e.g.,
                ``Semantic.XFORM_MAT4x4``). Alternative to ``dtype``/``shape``
                for specifying the attribute type.
            prim_mode: Prim binding mode (e.g., ``PrimMode.EXISTING_ONLY``).

        Returns:
            AttributeBinding[DLTensor] for scalar attribute writes.

        Example:
            ```python
            # String form — no imports needed for dtype specification:
            binding = renderer.bind_attribute(
                ["/World/Cube"], "xformOp:transform",
                dtype="float64", shape=(4, 4))

            # NumPy dtype objects also work:
            import numpy as np
            binding = renderer.bind_attribute(
                ["/World/Cube"], "visibility", dtype=np.int32)
            ```
        """
        return self.bind_attribute_async(
            prim_paths=prim_paths,
            attribute_name=attribute_name,
            dtype=dtype,
            shape=shape,
            semantic=semantic,
            prim_mode=prim_mode,
        ).wait()

    def bind_attribute_async(
        self,
        prim_paths: List[str],
        attribute_name: str,
        dtype: Any = None,
        shape: Optional[tuple] = None,
        semantic: Semantic = Semantic.NONE,
        prim_mode: PrimMode = PrimMode.EXISTING_ONLY,
    ) -> "Operation[AttributeBinding[DLTensor]]":
        """Create a persistent binding for scalar attribute writes (async).

        See :meth:`bind_attribute` for full documentation and examples.
        """
        return self._bind_attribute_internal(
            prim_paths=prim_paths,
            attribute_name=attribute_name,
            dtype=dtype,
            shape=shape,
            semantic=semantic,
            prim_mode=prim_mode,
            is_array=False,
        )

    def bind_array_attribute(
        self,
        prim_paths: List[str],
        attribute_name: str,
        dtype: Any = None,
        shape: Optional[tuple] = None,
        prim_mode: PrimMode = PrimMode.EXISTING_ONLY,
    ) -> "AttributeBinding[List[DLTensor]]":
        """Create a persistent binding for array attribute writes.

        Array attributes (e.g., ``float3[] points``) may have variable lengths
        per prim. The binding locks in the element dtype; subsequent writes can
        have varying tensor lengths but must use the same element type.

        Args:
            prim_paths: List of prim paths.
            attribute_name: Name of the array attribute.
            dtype: Element type. Accepts string names (``"float32"``,
                ``"float64"``, ``"int32"``, ``"token"``, ``"path"``),
                NumPy dtypes (``np.float32``, ``np.dtype("float64")``),
                any dtype object with ``.kind``/``.itemsize`` attributes
                (CuPy, JAX), Python built-ins (``float``, ``int``), or a
                ``DLDataType`` object.
            shape: Element shape tuple (e.g. ``(3,)`` for ``float3[]``).
                Mandatory for multi-component types when using numpy dtypes
                or string names. Ignored for string types and ``DLDataType``.
            prim_mode: Prim binding mode (e.g., ``PrimMode.CREATE_NEW``).

        Returns:
            AttributeBinding[List[DLTensor]] for array attribute writes.

        Example:
            ```python
            # String form — no imports needed for dtype specification:
            binding = renderer.bind_array_attribute(
                ["/World/Mesh"], "faceVertexCounts", dtype="int32")

            binding = renderer.bind_array_attribute(
                ["/World/Mesh"], "points", dtype="float32", shape=(3,))

            # NumPy dtype objects also work:
            import numpy as np
            binding = renderer.bind_array_attribute(
                ["/World/Mesh"], "normals", dtype=np.float32, shape=(3,))
            ```
        """
        return self.bind_array_attribute_async(
            prim_paths=prim_paths, attribute_name=attribute_name, dtype=dtype, shape=shape, prim_mode=prim_mode
        ).wait()

    def bind_array_attribute_async(
        self,
        prim_paths: List[str],
        attribute_name: str,
        dtype: Any = None,
        shape: Optional[tuple] = None,
        prim_mode: PrimMode = PrimMode.EXISTING_ONLY,
    ) -> "Operation[AttributeBinding[List[DLTensor]]]":
        """Create a persistent binding for array attribute writes (async).

        See :meth:`bind_array_attribute` for full documentation and examples.
        """
        return self._bind_attribute_internal(
            prim_paths=prim_paths,
            attribute_name=attribute_name,
            dtype=dtype,
            shape=shape,
            semantic=Semantic.NONE,
            prim_mode=prim_mode,
            is_array=True,
        )

    def _bind_attribute_internal(
        self,
        prim_paths: List[str],
        attribute_name: str,
        dtype: Any,
        shape: Optional[tuple],
        semantic: Semantic,
        prim_mode: PrimMode,
        is_array: bool,
    ) -> Operation[AttributeBinding]:
        """Internal: Create attribute binding (shared implementation for scalar and array)."""
        if self._handle is None:
            raise RuntimeError("Renderer is not valid")

        element_shape: Optional[tuple] = None

        if semantic != Semantic.NONE:
            # Explicit semantic: resolve dtype from the semantic constant
            if shape is not None:
                raise ValueError("shape= must not be provided when semantic= is specified")
            resolved_dtype = dtype if isinstance(dtype, DLDataType) else None
            semantic_constant, semantic_dtype = self._resolve_semantic_and_dtype(
                semantic, resolved_dtype, context="bind_attribute"
            )
        elif isinstance(dtype, DLDataType):
            # Explicit DLDataType: pass through directly with SEMANTIC_NONE
            semantic_constant = Semantic.NONE
            semantic_dtype = dtype
        elif dtype is not None:
            # New-style dtype + shape
            semantic_constant, semantic_dtype, element_shape = self._resolve_new_dtype_and_shape(
                dtype, shape, context="bind_attribute"
            )
        else:
            raise ValueError("bind_attribute: dtype is required. Use e.g. dtype='float32', shape=(3,)")

        binding_storage = _AttributeBindingDescStorage(
            prim_paths=prim_paths,
            attribute_name=attribute_name,
            dtype=semantic_dtype,
            semantic=semantic_constant,
            prim_mode=prim_mode,
            is_array=is_array,
        )

        result, handle = self._bindings.create_attribute_binding(self._handle, binding_storage.binding_desc)

        if result.status != bindings.OVRTX_API_SUCCESS:
            error_msg = self._bindings.get_last_error() or "Unknown error"
            raise RuntimeError(f"Failed to enqueue attribute binding creation: {error_msg}")

        binding_handle = AttributeBinding(
            handle.value, semantic_constant, semantic_dtype, self, is_array=is_array, shape=element_shape
        )

        op = Operation(
            renderer=self,
            op_id=result.op_index.value,
            handle=binding_handle,
            operation_name=f"bind_attribute({attribute_name})",
        )

        op._storage_refs = [binding_storage]

        return op

    def _unbind_attribute(self, binding: AttributeBinding) -> None:
        """Destroy a persistent attribute binding handle (synchronous - waits internally).

        Args:
            binding: AttributeBinding from `bind_attribute`

        Raises:
            RuntimeError: If renderer is invalid or destruction fails
        """
        self._unbind_attribute_async(binding).wait()

    def _unbind_attribute_async(self, binding: AttributeBinding) -> Operation[None]:
        """Destroy a persistent attribute binding handle asynchronously (returns Operation for manual control).

        Args:
            binding: AttributeBinding from `bind_attribute`

        Returns:
            Operation[None] for async control

        Raises:
            RuntimeError: If renderer is invalid or destruction fails
        """
        if self._handle is None:
            raise RuntimeError("Renderer is not valid")

        # Call C API
        result = self._bindings.destroy_attribute_binding(
            self._handle, bindings.ovrtx_attribute_binding_handle_t(binding.handle)
        )

        if result.status != bindings.OVRTX_API_SUCCESS:
            error_msg = self._bindings.get_last_error() or "Unknown error"
            raise RuntimeError(f"Failed to enqueue attribute binding destruction: {error_msg}")

        # Create operation
        op = Operation(
            renderer=self,
            op_id=result.op_index.value,
            handle=None,
            operation_name=f"unbind_attribute({binding.handle})",
        )

        return op

    def map_attribute(
        self,
        prim_paths: List[str],
        attribute_name: str,
        dtype: Any = None,
        shape: Optional[tuple] = None,
        semantic: Semantic = Semantic.NONE,
        device: Device = Device.CPU,
        device_id: int = 0,
        prim_mode: PrimMode = PrimMode.EXISTING_ONLY,
    ) -> AttributeMapping:
        """Map attribute buffer for direct writes (synchronous, by-name).

        Returns an AttributeMapping that provides access to an internal buffer.
        Write data using NumPy, Warp, or any ``__dlpack__``-compatible library,
        then call ``unmap_attribute()`` to apply the changes.

        When ``dtype``/``shape`` are provided, the returned tensor is
        automatically shaped to match. For example, ``dtype="float32",
        shape=(3,)`` returns a tensor with shape ``(N, 3)`` and ``float32``
        dtype, ready for direct NumPy/Warp consumption.

        Note:
            Array attributes (e.g., ``float3[] points``) are not supported for
            mapping. Use ``write_array_attribute()`` instead.

        Args:
            prim_paths: List of prim paths.
            attribute_name: Name of the attribute.
            dtype: Element type. Accepts string names (``"float32"``,
                ``"float64"``, ``"int32"``, ``"token"``, ``"path"``),
                NumPy dtypes (``np.float32``, ``np.dtype("float64")``),
                any dtype object with ``.kind``/``.itemsize`` attributes
                (CuPy, JAX), Python built-ins (``float``, ``int``), or a
                ``DLDataType`` object.
            shape: Element shape tuple (e.g. ``(3,)`` for 3-component vectors,
                ``(4, 4)`` for matrices). When provided, the mapped tensor is
                reshaped to ``(N, *shape)``. Ignored for ``DLDataType``.
            semantic: Optional attribute semantic (e.g.,
                ``Semantic.XFORM_MAT4x4``). Alternative to ``dtype``/``shape``
                for specifying the attribute type.
            device: Device for mapping (``Device.CPU`` or ``Device.CUDA``).
            device_id: Device ID (default 0, typically GPU index for CUDA).
            prim_mode: Prim binding mode (e.g., ``PrimMode.EXISTING_ONLY``).

        Returns:
            AttributeMapping with access to the internal buffer. When
            ``shape`` was provided, the tensor has dimensions ``(N, *shape)``
            with a scalar element dtype, ready for NumPy/Warp consumption.

        Raises:
            RuntimeError: If renderer is invalid or mapping fails.
            ValueError: If invalid parameters or dtype mismatch.

        Example:
            ```python
            import numpy as np

            # String form — no imports needed for dtype specification:
            mapping = renderer.map_attribute(
                ["/World/Cube"], "xformOp:transform",
                dtype="float64", shape=(4, 4))
            np.from_dlpack(mapping.tensor)[0] = np.eye(4)
            renderer.unmap_attribute(mapping)

            # NumPy dtype objects also work:
            with renderer.map_attribute(["/World/Cube"], "points",
                    dtype=np.float32, shape=(3,)) as mapping:
                np.from_dlpack(mapping.tensor)[:] = new_points
            ```
        """
        if self._handle is None:
            raise RuntimeError("Renderer is not valid")

        element_shape: Optional[tuple] = None

        if semantic != Semantic.NONE:
            # Explicit semantic: resolve dtype from the semantic constant
            if shape is not None:
                raise ValueError("shape= must not be provided when semantic= is specified")
            resolved_dtype = dtype if isinstance(dtype, DLDataType) else None
            semantic_constant, semantic_dtype = self._resolve_semantic_and_dtype(
                semantic, resolved_dtype, context="map_attribute"
            )
        elif isinstance(dtype, DLDataType):
            # Explicit DLDataType: pass through directly with SEMANTIC_NONE
            semantic_constant = Semantic.NONE
            semantic_dtype = dtype
        elif dtype is not None:
            # New-style dtype + shape
            semantic_constant, semantic_dtype, element_shape = self._resolve_new_dtype_and_shape(
                dtype, shape, context="map_attribute"
            )
        else:
            raise ValueError("map_attribute: dtype is required. Use e.g. dtype='float32', shape=(3,)")

        mapping_desc = bindings.ovrtx_mapping_desc_t(device_type=device, device_id=device_id)

        binding_storage = _AttributeBindingDescStorage(
            prim_paths=prim_paths,
            attribute_name=attribute_name,
            dtype=semantic_dtype,
            semantic=semantic_constant,
            prim_mode=prim_mode,
        )

        result, c_mapping = self._bindings.map_attribute(
            self._handle, binding_storage.binding_desc_or_handle, mapping_desc
        )

        if result.status != bindings.OVRTX_API_SUCCESS:
            error_msg = self._bindings.get_last_error() or "Unknown error"
            raise RuntimeError(f"Failed to map attribute: {error_msg}")

        reshaped_tensor = self._create_semantic_aware_dltensor(c_mapping.dl, semantic_constant, element_shape)

        mapping = AttributeMapping(
            mapping=c_mapping,
            renderer=self,
            dltensor=reshaped_tensor,
            binding_desc=binding_storage.binding_desc,
            device=device,
        )

        mapping._binding_storage = binding_storage

        return mapping

    def _map_attribute_by_binding(
        self, binding: AttributeBinding, device: Device = Device.CPU, device_id: int = 0
    ) -> AttributeMapping:
        """Map attribute buffer using a persistent binding handle (synchronous).

        Returns an AttributeMapping with access to the internal buffer.
        Write data using NumPy, Warp, or any ``__dlpack__``-compatible library,
        then call ``unmap_attribute()`` to apply the changes.

        When the binding was created with ``shape=``, the mapped tensor is
        automatically shaped to ``(N, *shape)`` with a scalar element dtype.

        Args:
            binding: AttributeBinding from bind_attribute().
            device: Device for mapping (Device.CPU or Device.CUDA).
            device_id: Device ID (default 0, typically GPU index for CUDA).

        Returns:
            AttributeMapping with access to the internal buffer.

        Raises:
            RuntimeError: If renderer is invalid or mapping fails.

        Example:
            ```python
            import numpy as np

            binding = renderer.bind_attribute(
                ["/World/Cube"], "xformOp:transform",
                dtype="float64", shape=(4, 4))
            mapping = binding.map()
            np.from_dlpack(mapping.tensor)[0] = np.eye(4)
            mapping.unmap()
            ```
        """
        if self._handle is None:
            raise RuntimeError("Renderer is not valid")

        mapping_desc = bindings.ovrtx_mapping_desc_t(device_type=device, device_id=device_id)

        binding_desc_or_handle = bindings.ovrtx_binding_desc_or_handle_t(
            binding_desc=bindings.ovrtx_binding_desc_t(),
            binding_handle=bindings.ovrtx_attribute_binding_handle_t(binding.handle),
        )

        result, c_mapping = self._bindings.map_attribute(self._handle, binding_desc_or_handle, mapping_desc)

        if result.status != bindings.OVRTX_API_SUCCESS:
            error_msg = self._bindings.get_last_error() or "Unknown error"
            raise RuntimeError(f"Failed to map attribute: {error_msg}")

        reshaped_tensor = self._create_semantic_aware_dltensor(c_mapping.dl, binding.semantic, binding.shape)

        mapping = AttributeMapping(
            mapping=c_mapping,
            renderer=self,
            dltensor=reshaped_tensor,
            binding_desc=None,
            device=device,
        )

        return mapping

    def unmap_attribute(
        self, mapping: AttributeMapping, event: Optional[int] = None, stream: Optional[int] = None
    ) -> None:
        """Unmap attribute and apply written data (synchronous - waits internally).

        Applies the data written to the mapped buffer to the stage representation
        and destroys the mapping. After this call, the mapping is invalid.

        For CUDA-mapped attributes, you can provide a CUDA event or stream handle
        to synchronize with ongoing GPU work before the data is consumed.

        Args:
            mapping: AttributeMapping from map_attribute()
            event: CUDA event handle to wait on before accessing data (from wp.Event.cuda_event).
                   The C library will wait for this event before reading the mapped data.
            stream: CUDA stream handle for synchronization (from wp.Stream.cuda_stream).
                    The C library will synchronize on this stream before reading the data.

        Raises:
            RuntimeError: If renderer is invalid or unmap fails
            ValueError: If event/stream provided for CPU-mapped attribute
            ValueError: If both event and stream provided (mutually exclusive)

        Note:
            Data must be fully written to mapping.tensor before calling this.
            The mapping cannot be used after unmap.
        """
        if mapping._device == Device.CPU and (event is not None or stream is not None):
            raise ValueError("CUDA sync parameters (event/stream) not applicable for CPU-mapped attributes")

        # Validation: event and stream are mutually exclusive
        if event is not None and stream is not None:
            raise ValueError("Cannot specify both event and stream; use one or the other")

        self.unmap_attribute_async(mapping, event=event, stream=stream).wait()
        mapping._unmapped = True

    def unmap_attribute_async(
        self, mapping: AttributeMapping, event: Optional[int] = None, stream: Optional[int] = None
    ) -> Operation[None]:
        """Unmap attribute asynchronously (returns Operation for manual control).

        Enqueues unmap operation and returns immediately. Use for advanced async control.

        For CUDA-mapped attributes, you can provide a CUDA event or stream handle
        to synchronize with ongoing GPU work before the data is consumed.

        Args:
            mapping: AttributeMapping from map_attribute()
            event: CUDA event handle to wait on before accessing data (from wp.Event.cuda_event)
            stream: CUDA stream handle for synchronization (from wp.Stream.cuda_stream)

        Returns:
            Operation[None] for async control

        Raises:
            RuntimeError: If renderer is invalid or unmap fails

        Note:
            After wait() completes, set mapping._unmapped = True manually.
            Validation of event/stream parameters should be done in the caller (unmap_attribute).
        """
        if self._handle is None:
            raise RuntimeError("Renderer is not valid")

        # Build CUDA sync struct (fields default to 0)
        cuda_sync = bindings.ovrtx_cuda_sync_t()
        if stream is not None:
            cuda_sync.stream = stream
        if event is not None:
            cuda_sync.wait_event = event

        # Call C API
        result = self._bindings.unmap_attribute(
            self._handle,
            bindings.ovrtx_map_handle_t(mapping.map_handle),
            cuda_sync,
        )

        if result.status != bindings.OVRTX_API_SUCCESS:
            error_msg = self._bindings.get_last_error() or "Unknown error"
            raise RuntimeError(f"Failed to enqueue attribute unmap: {error_msg}")

        # Create operation
        op = Operation(
            renderer=self,
            op_id=result.op_index.value,
            handle=None,
            operation_name=f"unmap_attribute({mapping.map_handle})",
        )

        return op

    def __bool__(self) -> bool:
        """Return True if renderer is valid, False if destroyed."""
        return self._handle is not None

    @property
    def config(self) -> RendererConfig:
        """Get the configuration used to create this renderer."""
        return self._config

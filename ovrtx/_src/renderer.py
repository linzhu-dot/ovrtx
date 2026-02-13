# Copyright (c) 2025-2026, NVIDIA CORPORATION. All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

import ctypes
import sys
from typing import Any, List, Optional

from . import bindings
from .dlpack import DLDataType, DLTensor
from .types import (
    AttributeBinding,
    AttributeMapping,
    FrameOutput,
    Operation,
    ProductOutput,
    RendererConfig,
    RendererResult,
    RenderProductSetOutputs,
    RenderVarOutput,
)


class _AttributeBindingDescStorage:
    """Helper class to keep binding descriptor and string arrays alive (private implementation detail)."""

    def __init__(
        self,
        prim_paths: List[str],
        attribute_name: str,
        dtype: DLDataType,
        semantic: int,
        prim_mode: str = "existing_only",
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

        # Convert prim_mode string to enum
        prim_mode_normalized = prim_mode.strip().lower()
        if prim_mode_normalized == "existing_only":
            prim_mode_enum = bindings.OVRTX_BINDING_PRIM_MODE_EXISTING_ONLY
        elif prim_mode_normalized == "must_exist":
            prim_mode_enum = bindings.OVRTX_BINDING_PRIM_MODE_MUST_EXIST
        elif prim_mode_normalized == "create_new":
            prim_mode_enum = bindings.OVRTX_BINDING_PRIM_MODE_CREATE_NEW
        else:
            raise ValueError(f"Invalid prim_mode: {prim_mode}. Must be 'existing_only', 'must_exist', or 'create_new'")

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
            prim_mode=bindings.ovrtx_binding_prim_mode_t(prim_mode_enum),
            flags=bindings.ovrtx_binding_flag_t(flags),
        )

        # Build binding_desc_or_handle (always use binding_desc, handle=0)
        self.binding_desc_or_handle = bindings.ovrtx_binding_desc_or_handle_t(
            binding_desc=self.binding_desc, binding_handle=0
        )


class _InputBufferStorage:
    """Helper class to keep input buffer and DLTensor arrays alive (private implementation detail)."""

    def __init__(self, dl_tensors: List[DLTensor], dirty_bits: Optional[bytes] = None):
        """Initialize input buffer storage.

        Args:
            dl_tensors: List of DLTensor objects (will be copied to array)
            dirty_bits: Optional dirty bit array (copied if provided)
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

        # Build input_buffer pointing to array
        self.input_buffer = bindings.ovrtx_input_buffer_t(
            tensors=self._tensor_array,  # Array reference kept in self
            tensor_count=len(self._tensor_storage),
            dirty_bits=dirty_bits_ptr,
            dirty_bits_size=dirty_bits_size,
            access_cuda_sync=bindings.ovrtx_cuda_sync_t(),
            done_cuda_sync=bindings.ovrtx_cuda_sync_t(),
        )


class Renderer:
    """High-level Pythonic renderer for OVRTX.

    Wraps the C library with automatic resource management. Resources are
    cleaned up automatically when the renderer goes out of scope.

    Example:
        ```python
        from ovrtx import Renderer, RendererConfig

        # Use defaults
        renderer = Renderer()

        # Or customize
        config = RendererConfig(mdl_base_path="custom/mdl/", sync_mode=True)
        renderer = Renderer(config=config)

        # Resources automatically cleaned up when renderer goes out of scope
        ```
    """

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

        Uses a whitelist of known config fields. Only whitelisted fields are converted.
        Only whitelisted fields are converted to C config entries.
        """
        # Whitelist: field_name -> factory_function
        # Field names match C keys exactly (see ovrtx_config.h)
        WHITELIST = {
            "sync_mode": bindings.ovrtx_renderer_config_entry_t.from_bool,
            "log_file_path": bindings.ovrtx_renderer_config_entry_t.from_string,
            "log_level": bindings.ovrtx_renderer_config_entry_t.from_string,
        }

        entries = []
        for field_name, factory_func in WHITELIST.items():
            value = getattr(config, field_name, None)
            if value is not None:
                entries.append(factory_func(field_name, value))

        return bindings.ovrtx_config_t(entries)

    def _resolve_semantic_and_dtype(
        self, semantic: Optional[str], dtype: Optional[DLDataType], context: str = "operation"
    ) -> tuple[int, DLDataType]:
        """Parse semantic string and resolve dtype.

        Semantics imply specific dtype requirements due to Fabric compatibility:
        - transform_4x4 → DLDataType(code=2, bits=64, lanes=16) (16 doubles)
        - color_rgba4b → DLDataType(code=1, bits=8, lanes=4) (4 uint8)
        - color_rgb3f → DLDataType(code=2, bits=32, lanes=3) (3 floats)

        Args:
            semantic: Semantic string ("transform_4x4", "color_rgba4b", "color_rgb3f", or None)
            dtype: User-provided dtype (optional if semantic implies dtype)
            context: Context string for error messages

        Returns:
            Tuple of (semantic_constant, semantic_dtype)

        Raises:
            ValueError: If semantic is invalid, dtype is required but not provided,
                       or if provided dtype doesn't match semantic
        """
        # 1. Parse semantic string to constant
        if semantic is None:
            semantic_constant = bindings.OVRTX_SEMANTIC_NONE
        else:
            semantic_normalized = semantic.strip().lower()
            if semantic_normalized in ("none", ""):
                semantic_constant = bindings.OVRTX_SEMANTIC_NONE
            elif semantic_normalized in ("transform_4x4", "transform", "matrix"):
                semantic_constant = bindings.OVRTX_SEMANTIC_TRANSFORM_4x4
            elif semantic_normalized in ("color_rgba4b", "rgba4b", "rgba"):
                semantic_constant = bindings.OVRTX_SEMANTIC_COLOR_RGBA4b
            elif semantic_normalized in ("color_rgb3f", "rgb3f", "rgb"):
                semantic_constant = bindings.OVRTX_SEMANTIC_COLOR_RGB3f
            else:
                raise ValueError(
                    f"Invalid semantic: {semantic}. "
                    f"Must be None, 'none', 'transform_4x4', 'color_rgba4b', 'color_rgb3f', etc."
                )

        # 2. Get inferred dtype for semantic (if any)
        if semantic_constant == bindings.OVRTX_SEMANTIC_TRANSFORM_4x4:
            inferred_dtype = DLDataType(code=2, bits=64, lanes=16)  # Matrix4d: 16 doubles
        elif semantic_constant == bindings.OVRTX_SEMANTIC_COLOR_RGBA4b:
            inferred_dtype = DLDataType(code=1, bits=8, lanes=4)  # RGBA: 4 uint8
        elif semantic_constant == bindings.OVRTX_SEMANTIC_COLOR_RGB3f:
            inferred_dtype = DLDataType(code=2, bits=32, lanes=3)  # RGB: 3 floats
        else:
            inferred_dtype = None  # NONE - user must provide dtype

        # 3. Resolve: use inferred, validate if both, or require user-provided
        if inferred_dtype is not None:
            # Use .value for code (DLDataTypeCode subclass), direct comparison for bits/lanes
            if dtype is not None and (
                dtype.code.value != inferred_dtype.code.value
                or dtype.bits != inferred_dtype.bits
                or dtype.lanes != inferred_dtype.lanes
            ):
                raise ValueError(
                    f"{context}: provided dtype (code={dtype.code}, bits={dtype.bits}, lanes={dtype.lanes}) "
                    f"doesn't match required dtype for semantic '{semantic}' "
                    f"(code={inferred_dtype.code}, bits={inferred_dtype.bits}, lanes={inferred_dtype.lanes})"
                )
            semantic_dtype = inferred_dtype
        elif dtype is None:
            raise ValueError(f"{context}: dtype is required when semantic is None")
        else:
            semantic_dtype = dtype

        return (semantic_constant, semantic_dtype)

    def _build_binding_desc(
        self,
        prim_paths: List[str],
        attribute_name: str,
        dtype: DLDataType,
        semantic: int,
        prim_mode: str = "existing_only",
        is_array: bool = False,
    ) -> _AttributeBindingDescStorage:
        """Construct binding descriptor from Python params.

        Args:
            prim_paths: List of prim paths
            attribute_name: Attribute name
            dtype: DLDataType for the attribute
            semantic: Semantic constant
            prim_mode: Prim mode string
            is_array: True for array attributes (e.g., float3[] points)

        Returns:
            _AttributeBindingDescStorage instance that keeps all C structures alive
        """
        return _AttributeBindingDescStorage(
            prim_paths=prim_paths,
            attribute_name=attribute_name,
            dtype=dtype,
            semantic=semantic,
            prim_mode=prim_mode,
            is_array=is_array,
        )

    def _build_input_buffer(
        self, dl_tensors: List[DLTensor], dirty_bits: Optional[bytes] = None
    ) -> _InputBufferStorage:
        """Construct input buffer from DLTensor list.

        Args:
            dl_tensors: List of DLTensor objects (one per prim for array attributes, single tensor otherwise)
            dirty_bits: Optional dirty bit array for selective updates

        Returns:
            _InputBufferStorage instance that keeps all C structures alive
        """
        return _InputBufferStorage(dl_tensors, dirty_bits=dirty_bits)

    def _build_mapping_desc(self, device: str, device_id: int = 0) -> bindings.ovrtx_mapping_desc_t:
        """Construct mapping descriptor from device string.

        Args:
            device: Device string ("cpu" or "cuda")
            device_id: Device ID (default 0, typically GPU index for CUDA)

        Returns:
            ovrtx_mapping_desc_t with device_type and device_id

        Raises:
            ValueError: If device string is invalid
        """
        device_normalized = device.strip().lower()
        if device_normalized == "cpu":
            device_type = 1  # kDLCPU
        elif device_normalized == "cuda":
            device_type = 2  # kDLCUDA
        else:
            raise ValueError(f"Invalid device: {device}. Must be 'cpu' or 'cuda'")

        return bindings.ovrtx_mapping_desc_t(device_type=device_type, device_id=device_id)

    def _create_semantic_aware_dltensor(self, original_dl_tensor: DLTensor, semantic: int) -> DLTensor:
        """Create semantic-aware reshaped DLTensor view optimized for NumPy users.

        Creates a new DLTensor with reshaped dimensions based on semantic, pointing to
        the same underlying memory (zero-copy view).

        Args:
            original_dl_tensor: Original DLTensor from C API (e.g., shape[N] lanes=16 for Matrix4d)
            semantic: Semantic constant (OVRTX_SEMANTIC_*)

        Returns:
            Reshaped DLTensor view (e.g., shape[N,4,4] lanes=1 for Matrix4d)
            For non-reshapable semantics, returns original tensor.

        Note:
            The returned DLTensor has _shape_storage attached to keep shape array alive.
        """
        # Get original shape
        if original_dl_tensor.ndim < 1 or original_dl_tensor.shape is None:
            return original_dl_tensor

        N = original_dl_tensor.shape[0]
        lanes = original_dl_tensor.dtype.lanes

        # Matrix4d: shape[N] lanes=16 → shape[N,4,4] lanes=1
        if semantic == bindings.OVRTX_SEMANTIC_TRANSFORM_4x4 and lanes == 16:
            # Create new DLTensor with reshaped dimensions
            reshaped = DLTensor()

            # Copy basic fields
            reshaped.data = original_dl_tensor.data
            reshaped.device = original_dl_tensor.device
            reshaped.byte_offset = original_dl_tensor.byte_offset
            reshaped.strides = None  # Contiguous

            # Reshape: (N,) lanes=16 → (N, 4, 4) lanes=1
            reshaped.ndim = 3
            shape_array = (ctypes.c_int64 * 3)(N, 4, 4)
            reshaped.shape = ctypes.cast(shape_array, ctypes.POINTER(ctypes.c_int64))

            # Update dtype to have lanes=1
            reshaped.dtype.code = original_dl_tensor.dtype.code
            reshaped.dtype.bits = original_dl_tensor.dtype.bits
            reshaped.dtype.lanes = 1

            # Attach shape array to keep it alive (Python attribute injection)
            reshaped._shape_storage = shape_array

            return reshaped

        # RGBA color: shape[N] lanes=4 → shape[N,4] lanes=1
        elif semantic == bindings.OVRTX_SEMANTIC_COLOR_RGBA4b and lanes == 4:
            reshaped = DLTensor()

            reshaped.data = original_dl_tensor.data
            reshaped.device = original_dl_tensor.device
            reshaped.byte_offset = original_dl_tensor.byte_offset
            reshaped.strides = None

            reshaped.ndim = 2
            shape_array = (ctypes.c_int64 * 2)(N, 4)
            reshaped.shape = ctypes.cast(shape_array, ctypes.POINTER(ctypes.c_int64))

            reshaped.dtype.code = original_dl_tensor.dtype.code
            reshaped.dtype.bits = original_dl_tensor.dtype.bits
            reshaped.dtype.lanes = 1

            reshaped._shape_storage = shape_array

            return reshaped

        # RGB color: shape[N] lanes=3 → shape[N,3] lanes=1
        elif semantic == bindings.OVRTX_SEMANTIC_COLOR_RGB3f and lanes == 3:
            reshaped = DLTensor()

            reshaped.data = original_dl_tensor.data
            reshaped.device = original_dl_tensor.device
            reshaped.byte_offset = original_dl_tensor.byte_offset
            reshaped.strides = None

            reshaped.ndim = 2
            shape_array = (ctypes.c_int64 * 2)(N, 3)
            reshaped.shape = ctypes.cast(shape_array, ctypes.POINTER(ctypes.c_int64))

            reshaped.dtype.code = original_dl_tensor.dtype.code
            reshaped.dtype.bits = original_dl_tensor.dtype.bits
            reshaped.dtype.lanes = 1

            reshaped._shape_storage = shape_array

            return reshaped

        # No reshaping needed for other semantics
        return original_dl_tensor

    def write_attribute(
        self,
        prim_paths: List[str],
        attribute_name: str,
        tensor: DLTensor,
        semantic: Optional[str] = None,
        dirty_bits: Optional[bytes] = None,
        prim_mode: str = "existing_only",
    ) -> None:
        """Write scalar attribute data synchronously.

        For repeated writes to the same attribute, consider using bind_attribute()
        followed by binding.write() for better performance.

        Args:
            prim_paths: List of prim paths to write to.
            attribute_name: Name of the attribute.
            tensor: Single DLTensor with one element per prim.
            semantic: Semantic string ("transform_4x4", "color_rgba4b", "color_rgb3f").
                If provided, dtype is inferred and validated against tensor's dtype.
            dirty_bits: Optional dirty bit array for selective updates.
            prim_mode: Prim mode ("existing_only", "must_exist", or "create_new").

        Raises:
            RuntimeError: If renderer is invalid or write fails.
            ValueError: If invalid parameters.

        Example:
            ```python
            matrix = Matrix4d()
            matrix.SetIdentity()
            renderer.write_attribute(
                prim_paths=["/World/Cube"],
                attribute_name="omni:fabric:worldMatrix",
                tensor=matrix.to_dltensor(),
                semantic="transform_4x4"
            )
            ```
        """
        self.write_attribute_async(
            prim_paths=prim_paths,
            attribute_name=attribute_name,
            tensor=tensor,
            semantic=semantic,
            dirty_bits=dirty_bits,
            prim_mode=prim_mode,
        ).wait()

    def write_array_attribute(
        self,
        prim_paths: List[str],
        attribute_name: str,
        tensors: List[DLTensor],
        dirty_bits: Optional[bytes] = None,
        prim_mode: str = "existing_only",
    ) -> None:
        """Write array attribute data synchronously.

        Array attributes (e.g., float3[] points) may have variable lengths per prim.
        Each tensor in the list corresponds to one prim.

        Args:
            prim_paths: List of prim paths to write to.
            attribute_name: Name of the array attribute.
            tensors: List of DLTensor, one per prim. Lengths may differ but
                element dtype must match the USD attribute schema. Must be CPU
                tensors; for GPU data, copy to CPU first (e.g., ``arr.numpy()``).
            dirty_bits: Optional dirty bit array for selective updates.
            prim_mode: Prim mode ("existing_only", "must_exist", or "create_new").

        Raises:
            RuntimeError: If renderer is invalid or write fails.
            ValueError: If invalid parameters or dtype mismatch.

        Note:
            Tensor dtype must exactly match the USD attribute's element type:

            - ``int[]`` → ``np.int32``
            - ``float3[]`` → ``np.float32`` with shape ``(N, 3)``
            - ``double3[]`` → ``np.float64`` with shape ``(N, 3)``

            Using the wrong dtype (e.g., numpy's default float64 for a float3
            attribute) will cause an element size mismatch error.

        Example:
            ```python
            from ovrtx._src.dlpack import DLTensor
            # int[] attribute - use int32
            face_counts = np.array([4, 4, 4], dtype=np.int32)
            renderer.write_array_attribute(
                prim_paths=["/World/Mesh"],
                attribute_name="faceVertexCounts",
                tensors=[DLTensor.from_dlpack(face_counts)]
            )
            # float3[] attribute - use float32 with shape (N, 3)
            points = np.array([[0, 0, 0], [1, 0, 0]], dtype=np.float32)
            renderer.write_array_attribute(
                prim_paths=["/World/Mesh"],
                attribute_name="points",
                tensors=[DLTensor.from_dlpack(points)]
            )
            ```
        """
        self.write_array_attribute_async(
            prim_paths=prim_paths,
            attribute_name=attribute_name,
            tensors=tensors,
            dirty_bits=dirty_bits,
            prim_mode=prim_mode,
        ).wait()

    def _write_attribute_by_binding(
        self, binding: AttributeBinding, tensors: List[DLTensor], dirty_bits: Optional[bytes] = None
    ) -> None:
        """Write attribute data using a binding handle synchronously (internal)."""
        self._write_attribute_by_binding_async(
            binding=binding, tensors=tensors, dirty_bits=dirty_bits, data_access="sync"
        ).wait()

    def write_attribute_async(
        self,
        prim_paths: List[str],
        attribute_name: str,
        tensor: DLTensor,
        semantic: Optional[str] = None,
        dirty_bits: Optional[bytes] = None,
        prim_mode: str = "existing_only",
    ) -> Operation[None]:
        """Write scalar attribute data asynchronously.

        Args:
            prim_paths: List of prim paths to write to.
            attribute_name: Name of the attribute.
            tensor: Single DLTensor with one element per prim.
            semantic: Semantic string (dtype inferred if provided).
            dirty_bits: Optional dirty bit array for selective updates.
            prim_mode: Prim mode string.

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
            data_access="sync",
        )

    def write_array_attribute_async(
        self,
        prim_paths: List[str],
        attribute_name: str,
        tensors: List[DLTensor],
        dirty_bits: Optional[bytes] = None,
        prim_mode: str = "existing_only",
    ) -> Operation[None]:
        """Write array attribute data asynchronously.

        Args:
            prim_paths: List of prim paths to write to.
            attribute_name: Name of the array attribute.
            tensors: List of DLTensor, one per prim. Element dtype must match the
                USD attribute schema. Must be CPU tensors; for GPU data, copy to
                CPU first (e.g., ``arr.numpy()``).
            dirty_bits: Optional dirty bit array for selective updates.
            prim_mode: Prim mode string.

        Returns:
            Operation for async control (yields None on completion).

        See Also:
            :meth:`write_array_attribute` for dtype matching requirements.
        """
        return self._write_attribute_internal(
            prim_paths=prim_paths,
            attribute_name=attribute_name,
            tensors=tensors,
            semantic=None,
            dirty_bits=dirty_bits,
            prim_mode=prim_mode,
            is_array=True,
            data_access="sync",
        )

    def _write_attribute_internal(
        self,
        prim_paths: List[str],
        attribute_name: str,
        tensors: List[DLTensor],
        semantic: Optional[str],
        dirty_bits: Optional[bytes],
        prim_mode: str,
        is_array: bool,
        data_access: str = "sync",
    ) -> Operation[None]:
        """Internal: Write attribute data (shared implementation for scalar and array)."""
        if self._handle is None:
            raise RuntimeError("Renderer is not valid")

        if not tensors:
            raise ValueError("tensors list cannot be empty")

        # Build input buffer from DLTensor list
        input_storage = self._build_input_buffer(tensors, dirty_bits=dirty_bits)

        # Extract dtype from first tensor and validate all tensors have same dtype
        tensor_dtype = tensors[0].dtype
        for i, t in enumerate(tensors[1:], start=1):
            if (
                t.dtype.code.value != tensor_dtype.code.value
                or t.dtype.bits != tensor_dtype.bits
                or t.dtype.lanes != tensor_dtype.lanes
            ):
                raise ValueError(
                    f"All tensors must have the same dtype. "
                    f"tensors[0] has dtype (code={tensor_dtype.code.value}, bits={tensor_dtype.bits}, lanes={tensor_dtype.lanes}), "
                    f"but tensors[{i}] has dtype (code={t.dtype.code.value}, bits={t.dtype.bits}, lanes={t.dtype.lanes})"
                )

        # Parse semantic and resolve/validate dtype against tensor's dtype
        semantic_constant, semantic_dtype = self._resolve_semantic_and_dtype(
            semantic, tensor_dtype, context="write_attribute"
        )

        # Build binding descriptor
        binding_storage = self._build_binding_desc(
            prim_paths=prim_paths,
            attribute_name=attribute_name,
            dtype=semantic_dtype,
            semantic=semantic_constant,
            prim_mode=prim_mode,
            is_array=is_array,
        )

        # Convert data_access string to enum
        if data_access == "sync":
            data_access_enum = bindings.OVRTX_DATA_ACCESS_SYNC
        elif data_access == "async":
            data_access_enum = bindings.OVRTX_DATA_ACCESS_ASYNC
        else:
            raise ValueError(f"Invalid data_access: {data_access}. Must be 'sync' or 'async'")

        # Call C API
        result = self._bindings.write_attribute(
            self._handle,
            binding_storage.binding_desc_or_handle,
            input_storage.input_buffer,
            bindings.ovrtx_data_access_t(data_access_enum),
        )

        if result.status != bindings.OVRTX_API_SUCCESS:
            error_msg = self._bindings.get_last_error() or "Unknown error"
            raise RuntimeError(f"Failed to enqueue attribute write: {error_msg}")

        # Create operation
        op = Operation(
            renderer=self, op_id=result.op_index.value, handle=None, operation_name=f"write_attribute({attribute_name})"
        )

        # For ASYNC access, keep storage alive by attaching to Operation
        if data_access == "async":
            op._storage_refs = [input_storage, binding_storage]

        return op

    def _write_attribute_by_binding_async(
        self,
        binding: AttributeBinding,
        tensors: List[DLTensor],
        dirty_bits: Optional[bytes] = None,
        data_access: str = "sync",
    ) -> Operation[None]:
        """Write attribute data using a binding handle asynchronously (returns Operation).

        Args:
            binding: AttributeBinding from bind_attribute()
            tensors: List of DLTensor objects
            dirty_bits: Optional dirty bit array for selective updates
            data_access: Data access mode ("sync" or "async")

        Returns:
            Operation for async control (yields None on completion)

        Raises:
            RuntimeError: If renderer is invalid or write fails
        """
        if self._handle is None:
            raise RuntimeError("Renderer is not valid")

        if not tensors:
            raise ValueError("tensors list cannot be empty")

        # Validate all tensors have same dtype
        first_dtype = tensors[0].dtype
        for i, t in enumerate(tensors[1:], start=1):
            if (
                t.dtype.code.value != first_dtype.code.value
                or t.dtype.bits != first_dtype.bits
                or t.dtype.lanes != first_dtype.lanes
            ):
                raise ValueError(
                    f"All tensors must have the same dtype. "
                    f"tensors[0] has dtype (code={first_dtype.code.value}, bits={first_dtype.bits}, lanes={first_dtype.lanes}), "
                    f"but tensors[{i}] has dtype (code={t.dtype.code.value}, bits={t.dtype.bits}, lanes={t.dtype.lanes})"
                )

        # Validate tensor dtype matches binding's expected dtype
        expected_dtype = binding.dtype
        if (
            first_dtype.code.value != expected_dtype.code.value
            or first_dtype.bits != expected_dtype.bits
            or first_dtype.lanes != expected_dtype.lanes
        ):
            raise ValueError(
                f"Tensor dtype (code={first_dtype.code.value}, bits={first_dtype.bits}, lanes={first_dtype.lanes}) "
                f"doesn't match binding's expected dtype "
                f"(code={expected_dtype.code.value}, bits={expected_dtype.bits}, lanes={expected_dtype.lanes})"
            )

        # Build input buffer from DLTensor list
        input_storage = self._build_input_buffer(tensors, dirty_bits=dirty_bits)

        # Use persistent handle
        binding_desc_or_handle = bindings.ovrtx_binding_desc_or_handle_t(
            binding_desc=bindings.ovrtx_binding_desc_t(),  # Empty descriptor
            binding_handle=bindings.ovrtx_attribute_binding_handle_t(binding.handle),
        )

        # Convert data_access string to enum
        if data_access == "sync":
            data_access_enum = bindings.OVRTX_DATA_ACCESS_SYNC
        elif data_access == "async":
            data_access_enum = bindings.OVRTX_DATA_ACCESS_ASYNC
        else:
            raise ValueError(f"Invalid data_access: {data_access}. Must be 'sync' or 'async'")

        # Call C API
        result = self._bindings.write_attribute(
            self._handle,
            binding_desc_or_handle,
            input_storage.input_buffer,
            bindings.ovrtx_data_access_t(data_access_enum),
        )

        if result.status != bindings.OVRTX_API_SUCCESS:
            error_msg = self._bindings.get_last_error() or "Unknown error"
            raise RuntimeError(f"Failed to enqueue attribute write: {error_msg}")

        # Create operation
        op = Operation(
            renderer=self,
            op_id=result.op_index.value,
            handle=None,
            operation_name=f"write_attribute(handle={binding.handle})",
        )

        # For ASYNC access, keep storage alive by attaching to Operation
        if data_access == "async":
            op._storage_refs = [input_storage]

        return op

    def bind_attribute(
        self,
        prim_paths: List[str],
        attribute_name: str,
        semantic: Optional[str] = None,
        dtype: Optional[DLDataType] = None,
        prim_mode: str = "existing_only",
    ) -> "AttributeBinding[DLTensor]":
        """Create a persistent binding for scalar attribute writes.

        Creates a binding handle that can be reused for multiple write() calls,
        avoiding the overhead of recreating the binding descriptor each time.

        Args:
            prim_paths: List of prim paths.
            attribute_name: Name of the attribute.
            semantic: Semantic string ("transform_4x4", "color_rgba4b", "color_rgb3f").
                If provided, dtype is inferred automatically.
            dtype: DLDataType for the attribute. Required only if semantic is None.
            prim_mode: Prim mode ("existing_only", "must_exist", or "create_new").

        Returns:
            AttributeBinding[DLTensor] for scalar attribute writes.

        Example:
            ```python
            binding = renderer.bind_attribute(
                prim_paths=["/World/Cube"],
                attribute_name="omni:fabric:worldMatrix",
                semantic="transform_4x4"
            )
            binding.write(matrix.to_dltensor())
            binding.unbind()
            ```
        """
        return self.bind_attribute_async(
            prim_paths=prim_paths, attribute_name=attribute_name, semantic=semantic, dtype=dtype, prim_mode=prim_mode
        ).wait()

    def bind_attribute_async(
        self,
        prim_paths: List[str],
        attribute_name: str,
        semantic: Optional[str] = None,
        dtype: Optional[DLDataType] = None,
        prim_mode: str = "existing_only",
    ) -> "Operation[AttributeBinding[DLTensor]]":
        """Create a persistent binding for scalar attribute writes (async)."""
        return self._bind_attribute_internal(
            prim_paths=prim_paths,
            attribute_name=attribute_name,
            semantic=semantic,
            dtype=dtype,
            prim_mode=prim_mode,
            is_array=False,
        )

    def bind_array_attribute(
        self,
        prim_paths: List[str],
        attribute_name: str,
        dtype: Optional[DLDataType] = None,
        prim_mode: str = "existing_only",
    ) -> "AttributeBinding[List[DLTensor]]":
        """Create a persistent binding for array attribute writes.

        Array attributes (e.g., float3[] points) may have variable lengths per prim.
        The binding locks in the element dtype; subsequent writes can have varying
        tensor lengths but must use the same element type.

        Args:
            prim_paths: List of prim paths.
            attribute_name: Name of the array attribute.
            dtype: DLDataType for the attribute elements. Must match the USD
                attribute schema (e.g., int32 for ``int[]``, float32 for ``float3[]``).
            prim_mode: Prim mode ("existing_only", "must_exist", or "create_new").

        Returns:
            AttributeBinding[List[DLTensor]] for array attribute writes.

        Note:
            The dtype specifies the element type, not the array length. Each
            tensor in subsequent writes can have different lengths, but all must
            share the same element dtype matching the USD schema.

        Example:
            ```python
            from ovrtx._src.dlpack import DLDataType
            # int[] attribute binding
            binding = renderer.bind_array_attribute(
                prim_paths=["/World/Mesh1", "/World/Mesh2"],
                attribute_name="faceVertexCounts",
                dtype=DLDataType.from_str("int32"),
            )
            binding.write([counts1, counts2])  # Variable lengths OK
            binding.unbind()
            ```
        """
        return self.bind_array_attribute_async(
            prim_paths=prim_paths, attribute_name=attribute_name, dtype=dtype, prim_mode=prim_mode
        ).wait()

    def bind_array_attribute_async(
        self,
        prim_paths: List[str],
        attribute_name: str,
        dtype: Optional[DLDataType] = None,
        prim_mode: str = "existing_only",
    ) -> "Operation[AttributeBinding[List[DLTensor]]]":
        """Create a persistent binding for array attribute writes (async)."""
        return self._bind_attribute_internal(
            prim_paths=prim_paths,
            attribute_name=attribute_name,
            semantic=None,
            dtype=dtype,
            prim_mode=prim_mode,
            is_array=True,
        )

    def _bind_attribute_internal(
        self,
        prim_paths: List[str],
        attribute_name: str,
        semantic: Optional[str],
        dtype: Optional[DLDataType],
        prim_mode: str,
        is_array: bool,
    ) -> Operation[AttributeBinding]:
        """Internal: Create attribute binding (shared implementation for scalar and array)."""
        if self._handle is None:
            raise RuntimeError("Renderer is not valid")

        # Parse semantic and resolve dtype
        semantic_constant, semantic_dtype = self._resolve_semantic_and_dtype(semantic, dtype, context="bind_attribute")

        # Build binding descriptor
        binding_storage = self._build_binding_desc(
            prim_paths=prim_paths,
            attribute_name=attribute_name,
            dtype=semantic_dtype,
            semantic=semantic_constant,
            prim_mode=prim_mode,
            is_array=is_array,
        )

        # Call C API
        result, handle = self._bindings.create_attribute_binding(self._handle, binding_storage.binding_desc)

        if result.status != bindings.OVRTX_API_SUCCESS:
            error_msg = self._bindings.get_last_error() or "Unknown error"
            raise RuntimeError(f"Failed to enqueue attribute binding creation: {error_msg}")

        # Create AttributeBinding with raw handle value, semantic, dtype, renderer reference, and is_array flag
        binding_handle = AttributeBinding(handle.value, semantic_constant, semantic_dtype, self, is_array=is_array)

        # Create operation
        op = Operation(
            renderer=self,
            op_id=result.op_index.value,
            handle=binding_handle,
            operation_name=f"bind_attribute({attribute_name})",
        )

        # Store binding_storage to keep it alive until operation completes
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
        semantic: Optional[str] = None,
        dtype: Optional[DLDataType] = None,
        device: str = "cpu",
        device_id: int = 0,
        prim_mode: str = "existing_only",
    ) -> AttributeMapping:
        """Map attribute buffer for direct zero-copy writes (synchronous, by-name).

        Returns an AttributeMapping that provides direct access to RTX's internal buffer.
        Write data via the DLPack protocol (NumPy, Warp, etc.), then call unmap_attribute()
        to apply the changes.

        For repeated operations on the same attribute, consider using bind_attribute()
        followed by binding.map() for better performance.

        Note:
            Array attributes (e.g., float3[] points) are not supported for mapping.
            They may have variable lengths per prim, which is incompatible with
            contiguous tensor mapping. Use write_array_attribute() instead.

        Args:
            prim_paths: List of prim paths.
            attribute_name: Name of the attribute.
            semantic: Semantic string ("transform_4x4", "color_rgba4b", "color_rgb3f").
                If provided, dtype is inferred automatically.
            dtype: DLDataType for the attribute. Required only if semantic is None.
            device: Device for mapping ("cpu" or "cuda").
            device_id: Device ID (default 0, typically GPU index for CUDA).
            prim_mode: Prim mode ("existing_only", "must_exist", or "create_new").

        Returns:
            AttributeMapping with zero-copy access to RTX's internal buffer.
            - For Matrix4d: tensor has shape (N, 4, 4) for NumPy-friendly operations
            - For RGBA color: tensor has shape (N, 4) for RGBA channels
            - For RGB color: tensor has shape (N, 3) for RGB channels

        Raises:
            RuntimeError: If renderer is invalid or mapping fails.
            ValueError: If invalid parameters or dtype mismatch.

        Example:
            ```python
            import numpy as np
            mapping = renderer.map_attribute(
                prim_paths=["/World/Cube"],
                attribute_name="omni:fabric:worldMatrix",
                semantic="transform_4x4"
            )
            np.from_dlpack(mapping.tensor)[0] = np.eye(4)
            renderer.unmap_attribute(mapping)
            ```

        Or use as context manager:
            ```python
            with renderer.map_attribute(...) as mapping:
                np.from_dlpack(mapping.tensor)[0] = np.eye(4)
            ```
        """
        if self._handle is None:
            raise RuntimeError("Renderer is not valid")

        # Parse semantic and resolve dtype
        semantic_constant, semantic_dtype = self._resolve_semantic_and_dtype(semantic, dtype, context="map_attribute")

        # Build mapping descriptor
        mapping_desc = self._build_mapping_desc(device, device_id)

        # Build inline descriptor
        binding_storage = self._build_binding_desc(
            prim_paths=prim_paths,
            attribute_name=attribute_name,
            dtype=semantic_dtype,
            semantic=semantic_constant,
            prim_mode=prim_mode,
        )

        # Call C API (synchronous)
        result, c_mapping = self._bindings.map_attribute(
            self._handle, binding_storage.binding_desc_or_handle, mapping_desc
        )

        if result.status != bindings.OVRTX_API_SUCCESS:
            error_msg = self._bindings.get_last_error() or "Unknown error"
            raise RuntimeError(f"Failed to map attribute: {error_msg}")

        # Create semantic-aware reshaped DLTensor view
        reshaped_tensor = self._create_semantic_aware_dltensor(c_mapping.dl, semantic_constant)

        # Create AttributeMapping wrapper
        mapping = AttributeMapping(
            mapping=c_mapping,
            renderer=self,
            dltensor=reshaped_tensor,
            binding_desc=binding_storage.binding_desc,
            device=device,
        )

        # Keep binding storage alive by attaching to mapping
        mapping._binding_storage = binding_storage

        return mapping

    def _map_attribute_by_binding(
        self, binding: AttributeBinding, device: str = "cpu", device_id: int = 0
    ) -> AttributeMapping:
        """Map attribute buffer using a persistent binding handle (synchronous).

        Returns an AttributeMapping that provides direct access to RTX's internal buffer.
        Write data via the DLPack protocol (NumPy, Warp, etc.), then call unmap_attribute()
        to apply the changes.

        This is more efficient than map_attribute() for repeated operations since the
        binding is already created.

        Args:
            binding: AttributeBinding from bind_attribute()
            device: Device for mapping ("cpu" or "cuda")
            device_id: Device ID (default 0, typically GPU index for CUDA)

        Returns:
            AttributeMapping with zero-copy access to RTX's internal buffer.
            Tensor shape is determined by the semantic stored in the binding:
            - For Matrix4d: tensor has shape (N, 4, 4) for NumPy-friendly operations
            - For RGBA color: tensor has shape (N, 4) for RGBA channels
            - For RGB color: tensor has shape (N, 3) for RGB channels

        Raises:
            RuntimeError: If renderer is invalid or mapping fails

        Example:
            ```python
            import numpy as np
            # Create binding once
            binding = renderer.bind_attribute(
                prim_paths=["/World/Cube"],
                attribute_name="omni:fabric:worldMatrix",
                semantic="transform_4x4"
            )
            # Map using binding (faster for repeated operations)
            mapping = binding.map()
            np.from_dlpack(mapping.tensor)[0] = np.eye(4)
            mapping.unmap()
            ```
        """
        if self._handle is None:
            raise RuntimeError("Renderer is not valid")

        # Build mapping descriptor
        mapping_desc = self._build_mapping_desc(device, device_id)

        # Use persistent handle
        binding_desc_or_handle = bindings.ovrtx_binding_desc_or_handle_t(
            binding_desc=bindings.ovrtx_binding_desc_t(),  # Empty descriptor
            binding_handle=bindings.ovrtx_attribute_binding_handle_t(binding.handle),
        )

        # Call C API (synchronous)
        result, c_mapping = self._bindings.map_attribute(self._handle, binding_desc_or_handle, mapping_desc)

        if result.status != bindings.OVRTX_API_SUCCESS:
            error_msg = self._bindings.get_last_error() or "Unknown error"
            raise RuntimeError(f"Failed to map attribute: {error_msg}")

        # Create semantic-aware reshaped DLTensor view (using semantic from binding)
        reshaped_tensor = self._create_semantic_aware_dltensor(c_mapping.dl, binding.semantic)

        # Create AttributeMapping wrapper
        mapping = AttributeMapping(
            mapping=c_mapping,
            renderer=self,
            dltensor=reshaped_tensor,
            binding_desc=None,  # No inline descriptor when using binding handle
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
        # Validation: CUDA sync only for CUDA-mapped attributes
        if mapping._device == "cpu" and (event is not None or stream is not None):
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

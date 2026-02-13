# Copyright (c) 2025-2026, NVIDIA CORPORATION. All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

import sys
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Generic, Optional, TypeVar

from .dlpack import DLDataType, DLTensor, ManagedDLTensor

if TYPE_CHECKING:
    from .renderer import Renderer

T = TypeVar("T")
_BindingTensorT = TypeVar("_BindingTensorT")


class Operation(Generic[T]):
    """Represents an enqueued async operation.

    Call wait() to block until the operation completes.
    """

    @dataclass
    class _Result(Generic[T]):
        """Internal result state.

        Status is inferred from value and errors:
        - errors present → failed
        - value is None, no errors → timeout (pending)
        - value present, no errors → completed
        """

        value: Optional[T]
        errors: list[str]

        @property
        def is_resolved(self) -> bool:
            """True if resolved (succeeded or failed), False if pending."""
            return self.value is not None or bool(self.errors)

    def __init__(
        self,
        renderer: "Renderer",
        op_id: int,
        handle: Optional[T],
        operation_name: str,
    ):
        """Initialize operation (internal use - created by Renderer methods)."""
        self._renderer = renderer
        self._op_id = op_id
        self._handle = handle
        self._operation_name = operation_name
        self._result: Optional[Operation._Result[T]] = None

    @property
    def op_id(self) -> int:
        """The operation ID."""
        return self._op_id

    @property
    def handle(self) -> Optional[T]:
        """The operation-specific handle."""
        return self._handle

    def wait(self, timeout_ns: Optional[int] = None) -> Optional[T]:
        """Wait for operation to complete.

        Args:
            timeout_ns: Timeout in nanoseconds.
                       None (default) = infinite wait
                        0 = non-blocking poll, returns immediately
                       >0 = wait for specified duration

        Returns:
            The operation result value if completed successfully.
            None if timed out (operation still pending).

        Raises:
            RuntimeError: If operation failed.

        Note:
            This API assumes operations always return non-None values on success,
            which aligns with the C API design where operations return handles.
            Therefore, None always unambiguously indicates timeout.

        Examples:
            ```python
            # Infinite wait (never returns None)
            handle = op.wait()
            assert handle is not None

            # Poll without blocking
            handle = op.wait(timeout_ns=0)
            if handle is None:
                print("Not ready yet")

            # Custom timeout
            handle = op.wait(timeout_ns=5_000_000_000)  # 5 seconds
            if handle is None:
                print("Timed out")
            ```
        """
        if self._result is None or not self._result.is_resolved:
            self._result = self._renderer._wait_operation(self, timeout_ns)

        result = self._result

        # Failed - has errors
        if result.errors:
            error_msg = "\n  ".join(result.errors)
            raise RuntimeError(f"Operation '{self._operation_name}' failed:\n  {error_msg}")

        # Success or timeout - returned value is None on timeout
        return result.value

    def __repr__(self) -> str:
        if self._result is None:
            return f"Operation({self._operation_name}, op_id={self._op_id}, pending)"
        if self._result.errors:
            status = "failed"
        else:
            status = "pending" if self._result.value is None else "completed"
        return f"Operation({self._operation_name}, op_id={self._op_id}, {status})"


class _MappedRenderVar:
    """Context manager for mapped render variable.

    Wraps a RenderVarOutput that has been mapped, providing automatic
    cleanup when the context exits. Supports optional CUDA synchronization
    on unmap for GPU pipelines.

    Usage:
        with render_var.map(device="cuda") as mapping:
            wp_image = wp.from_dlpack(mapping.tensor)
            stream = wp.Stream(device=wp_image.device)
            # ... GPU work on stream ...
            mapping.unmap(stream=stream.cuda_stream)  # Optional explicit sync
    """

    def __init__(self, render_var: "RenderVarOutput", device: str = "cpu"):
        """Initialize context manager.

        Args:
            render_var: Parent RenderVarOutput (must be already mapped)
            device: Device type ("cpu" or "cuda")
        """
        self._render_var = render_var
        self._managed_tensor: Optional[ManagedDLTensor] = None
        self._device = device
        self._unmapped = False

    @property
    def device(self) -> str:
        """Device type this output is mapped to ('cpu' or 'cuda')."""
        return self._device

    @property
    def tensor(self) -> ManagedDLTensor:
        """Get ManagedDLTensor for DLPack interop (NumPy, Warp, PyTorch, etc.).

        Raises:
            RuntimeError: If accessed before entering context manager or after unmap().
        """
        if self._managed_tensor is None:
            raise RuntimeError("Mapping not entered - use within 'with' statement")
        if self._unmapped:
            raise RuntimeError("Mapping already unmapped - tensor no longer valid")
        return self._managed_tensor

    def unmap(self, event: Optional[int] = None, stream: Optional[int] = None) -> None:
        """Unmap render output with optional CUDA synchronization.

        For CUDA-mapped outputs, you can specify an event or stream that the
        C library will wait on before reclaiming the buffer. This ensures your
        GPU work completes before the buffer is reused.

        Args:
            event: CUDA event handle to wait on (from wp.Event.cuda_event or stream.record_event().cuda_event)
            stream: CUDA stream handle to synchronize with (from wp.Stream.cuda_stream)

        Raises:
            ValueError: If event/stream provided for CPU-mapped output
            ValueError: If both event and stream provided (mutually exclusive)

        Note:
            If called within a context manager, __exit__ becomes a no-op.
            If not called, context manager's __exit__ unmaps without CUDA sync.
        """
        if self._unmapped:
            return

        # Validation: CUDA sync only for CUDA-mapped outputs
        if self._device == "cpu" and (event is not None or stream is not None):
            raise ValueError("CUDA sync parameters (event/stream) not applicable for CPU-mapped outputs")

        # Validation: event and stream are mutually exclusive
        if event is not None and stream is not None:
            raise ValueError("Cannot specify both event and stream; use one or the other")

        # Build cuda_sync and call existing _unmap_output
        cuda_sync = None
        if event is not None or stream is not None:
            from . import bindings

            cuda_sync = bindings.ovrtx_cuda_sync_t()
            if stream is not None:
                cuda_sync.stream = stream
            if event is not None:
                cuda_sync.wait_event = event

        rv = self._render_var
        rv._renderer._unmap_output(rv._map_handle, cuda_sync)
        rv._tensor = None
        rv._map_handle = None
        self._managed_tensor = None
        self._unmapped = True

    def __enter__(self) -> "_MappedRenderVar":
        """Context manager entry - returns self for access to tensor and unmap()."""
        self._managed_tensor = ManagedDLTensor(
            dl_tensor=self._render_var._tensor,
            manager_ctx=self._render_var,
            deleter_callback=None,
            readonly=True,
        )
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - cleanup mapped resources if not already unmapped."""
        if not self._unmapped:
            self.unmap()
        return False


class RenderVarOutput(Generic[T]):
    """Single render variable output with mapping capabilities.

    Provides access to pixel data via map() and context manager.
    Supports DLPack protocol for interop (NumPy, Warp, PyTorch, etc.).

    Usage:
        # CPU mapping (default)
        with render_var.map() as mapping:
            np_array = np.from_dlpack(mapping.tensor)

        # Explicit CPU mapping
        with render_var.map(device="cpu") as mapping:
            np_array = np.from_dlpack(mapping.tensor)

        # CUDA device mapping
        with render_var.map(device="cuda") as mapping:
            wp_array = wp.from_dlpack(mapping.tensor)
            wp.launch(my_kernel, inputs=[wp_array])
    """

    def __init__(
        self,
        name: str,
        handle: T,
        renderer: "Renderer",
    ):
        """Internal: Created by Renderer._fetch_results()."""
        self.name = name
        self.handle = handle
        self._renderer = renderer
        self._tensor: Optional[DLTensor] = None
        self._map_handle: Any = None

    def map(self, device: str = "cpu", sync_stream: Optional[int] = None) -> "_MappedRenderVar":
        """Map render variable to device memory for access.

        Args:
            device: Target device - "cpu" or "cuda" (case-insensitive).
                   - "cpu": Map to host memory
                   - "cuda": Map to CUDA device memory (linear buffer)
            sync_stream: CUDA stream handle for render completion sync. The provided stream
                will wait for render to complete before any subsequent work executes.
                Default: 1 for CUDA (default stream), 0 for CPU.

        Returns:
            Context manager with tensor property and optional unmap() for CUDA sync

        Raises:
            RuntimeError: If mapping fails or already mapped
            ValueError: If device parameter is invalid

        Note:
            The buffer is ready for immediate use when map() returns. The buffer
            remains valid until the context manager exits or unmap() is called.
        """
        from .bindings import OVRTX_MAP_DEVICE_TYPE_CPU, OVRTX_MAP_DEVICE_TYPE_CUDA

        if (device_lower := device.strip().lower()) == "cuda":
            device_type = OVRTX_MAP_DEVICE_TYPE_CUDA
        elif not device or device_lower == "cpu":
            device_type = OVRTX_MAP_DEVICE_TYPE_CPU
        else:
            raise ValueError(f"Unsupported device: {device}")

        if self._tensor is not None:
            raise RuntimeError(f"Render var '{self.name}' already mapped")

        self._tensor, self._map_handle = self._renderer._map_output(
            self.handle, device_type=device_type, sync_stream=sync_stream
        )

        # Return context manager wrapper with device for validation
        return _MappedRenderVar(self, device=device_lower)

    def release(self) -> None:
        """Release mapped resources.

        Safe to call multiple times. Call this after completing all
        operations on exported DLPack tensors for deterministic cleanup.
        """
        if self._tensor is not None and self._map_handle is not None:
            self._renderer._unmap_output(self._map_handle)
            self._tensor = None
            self._map_handle = None

    def __repr__(self) -> str:
        status = "mapped" if self._tensor else "unmapped"
        return f"RenderVarOutput(name='{self.name}', {status})"


@dataclass
class FrameOutput(Generic[T]):
    """Single frame with multiple render variables."""

    start_time: float
    end_time: float
    render_vars: dict[str, RenderVarOutput[T]]


@dataclass
class ProductOutput(Generic[T]):
    """Single render product with multiple frames."""

    name: str
    frames: list[FrameOutput[T]]


class RenderProductSetOutputs(Generic[T]):
    """Dict-like container for rendering results from a step operation.

    Acts as a dictionary mapping render product paths to ProductOutput instances.
    Auto-destroys resources when it goes out of scope.

    Usage:
        # Dict-like iteration
        products = renderer.step(...)
        for product_name, product in products.items():
            for frame in product.frames:
                for var_name, render_var in frame.render_vars.items():
                    with render_var.map() as tensor:
                        # Process tensor...

        # Dict-like indexing
        product = products["/Render/Product0"]

        # Membership test
        if "/Render/Product0" in products:
            ...

        # Auto-cleanup via context manager
        with products as ctx:
            for name, product in ctx.items():
                ...
    """

    def __init__(
        self,
        operation: Operation[T],
        products: dict[str, ProductOutput[T]],
    ):
        """Internal: Created by Renderer._fetch_results().

        Args:
            operation: The step operation (provides access to renderer and handle)
            products: Parsed product outputs keyed by render product name
        """
        self._operation = operation
        self._outputs = products
        self._destroyed = False

    # Dict-like protocol
    def __getitem__(self, key: str) -> ProductOutput[T]:
        """Get product by render product path."""
        return self._outputs[key]

    def __iter__(self):
        """Iterate over render product paths."""
        return iter(self._outputs)

    def __len__(self) -> int:
        """Number of render products."""
        return len(self._outputs)

    def __contains__(self, key: str) -> bool:
        """Check if render product path exists."""
        return key in self._outputs

    def keys(self):
        """Get all render product paths."""
        return self._outputs.keys()

    def values(self):
        """Get all ProductOutput instances."""
        return self._outputs.values()

    def items(self):
        """Get all (path, ProductOutput) pairs."""
        return self._outputs.items()

    def __enter__(self):
        """Support context manager protocol."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Auto-cleanup on context exit."""
        self.destroy()
        return False

    def destroy(self):
        """Explicitly destroy resources.

        Frees C-side metadata. Safe to call multiple times.
        Automatically called on garbage collection or context exit.
        """
        if not self._destroyed:
            self._operation._renderer._destroy_results(self._operation)
            self._destroyed = True

    def __del__(self):
        """Auto-cleanup on garbage collection."""
        if not self._destroyed:
            try:
                self.destroy()
            except Exception as e:
                print(f"Warning: Exception during RenderProductSetOutputs cleanup in __del__: {e}", file=sys.stderr)
            finally:
                self._destroyed = True

    def __repr__(self) -> str:
        if self._destroyed:
            return "RenderProductSetOutputs(destroyed)"
        return f"RenderProductSetOutputs({list(self._outputs.keys())})"


class RendererResult(Generic[T]):
    """Result of a renderer step operation.

    Provides access to rendering metadata via wait().

    Usage:
        ```python
        result = renderer.step_async(...)
        metadata = result.wait()  # Block until ready

        # Or poll without blocking
        metadata = result.wait(step_timeout_ns=0)
        if metadata is None:
            print("Not ready yet")
        ```
    """

    def __init__(self, operation: Operation[T]):
        """Internal: Created by Renderer.step_async().

        Args:
            operation: The step operation with renderer-specific handle
        """
        self._operation = operation
        self._metadata: Optional[RenderProductSetOutputs[T]] = None

    @property
    def step_complete(self) -> bool:
        """Check if step operation has completed.

        Useful for diagnosing which stage timed out when wait() returns None.

        Returns:
            True if step completed (regardless of fetch status), False otherwise.

        Example:
            ```python
            metadata = result.wait(step_timeout_ns=1000, fetch_timeout_ns=500)
            if metadata is None:
                if not result.step_complete:
                    print("Step timed out - increase step_timeout_ns")
                else:
                    print("Fetch timed out - increase fetch_timeout_ns")
            ```
        """
        r = self._operation._result
        return r is not None and r.is_resolved

    def wait(
        self, step_timeout_ns: Optional[int] = None, fetch_timeout_ns: Optional[int] = None
    ) -> Optional[RenderProductSetOutputs[T]]:
        """Wait for step operation and fetch rendering results.

        Two-stage operation with independent timeout control:

        1. Step operation: Raytracing/GPU work
            - If times out, returns None immediately without attempting fetch
        2. Fetch operation: Memory transfer (only if step completed)
            - If times out, returns None

        Args:
            step_timeout_ns: Timeout for step operation.
                            None = infinite (default), 0 = poll, >0 = wait duration
            fetch_timeout_ns: Timeout for fetch operation.
                             None = infinite (default), 0 = poll, >0 = wait duration

        Returns:
            RenderProductSetOutputs if both stages complete, None if either times out.

            The returned results auto-destroy their C resources when they
            go out of scope. Can be used as context manager for explicit control.
            Returns cached results on subsequent calls (idempotent).

        Raises:
            RuntimeError: If step or fetch operation failed.

        Examples:
            ```python
            # Infinite wait (most common - never returns None)
            metadata = result.wait()
            assert metadata is not None

            # Poll step without blocking (returns immediately)
            metadata = result.wait(step_timeout_ns=0)
            if metadata is None:
                print("Step not ready yet")  # Fetch never attempted

            # Custom timeouts for both stages
            metadata = result.wait(
                step_timeout_ns=5_000_000_000,  # 5 seconds for raytracing
                fetch_timeout_ns=100_000_000    # 100ms for memory transfer
            )
            if metadata is None:
                # Use step_complete property to diagnose which stage timed out
                if not result.step_complete:
                    print("Step timed out")
                else:
                    print("Fetch timed out (step complete)")
            ```
        """
        # Return cached if already fetched
        if self._metadata is not None:
            return self._metadata

        # Stage 1: Wait for step operation (raytracing/GPU work)
        handle = self._operation.wait(step_timeout_ns)

        # Early return if step timed out - don't attempt fetch
        if handle is None:
            return None

        # Stage 2: Fetch results (memory transfer if needed)
        try:
            self._metadata = self._operation._renderer._fetch_results(
                operation=self._operation, timeout_ns=fetch_timeout_ns
            )
        except Exception as e:
            raise RuntimeError(f"Failed to fetch rendering results: {e}") from e

        return self._metadata

    def __del__(self):
        """Auto-fetch on garbage collection to prevent C resource leaks.

        If the step operation completes but results are never fetched, C resources
        will leak and trigger error logs. This safety net ensures cleanup happens.
        """
        r = self._operation._result
        if self._metadata is None and r is not None and r.is_resolved:
            try:
                # Step completed but never fetched - try to clean up
                self.wait()
            except Exception:
                pass  # Suppress all exceptions in __del__

    def __repr__(self) -> str:
        if self._metadata is not None:
            return f"RendererResult(op_id={self._operation.op_id}, fetched)"

        if self._operation._result is None:
            return f"RendererResult(op_id={self._operation.op_id}, step_pending)"
        if self._operation._result.errors:
            status = "failed"
        else:
            status = "step_complete" if self._operation._result.value is not None else "step_pending"
        return f"RendererResult(op_id={self._operation.op_id}, {status})"


class AttributeBinding(Generic[_BindingTensorT]):
    """Persistent attribute binding for efficient repeated writes or maps.

    Created by Renderer.bind_attribute() or Renderer.bind_array_attribute().
    Reuse for multiple write operations to avoid recreating the binding descriptor.

    The type parameter indicates the expected data type for write():
    - AttributeBinding[DLTensor]: scalar attributes (one value per prim)
    - AttributeBinding[list[DLTensor]]: array attributes (variable-length per prim)

    Example (scalar):
        ```python
        binding = renderer.bind_attribute(
            prim_paths=["/World/Cube"],
            attribute_name="xformOp:transform",
            semantic="transform_4x4"
        )
        binding.write(frame1.to_dltensor())
        binding.write(frame2.to_dltensor())
        binding.unbind()
        ```

    Example (array):
        ```python
        binding = renderer.bind_array_attribute(
            prim_paths=["/World/Mesh1", "/World/Mesh2"],
            attribute_name="faceVertexCounts"
        )
        binding.write([counts1, counts2])  # List of tensors, one per prim
        binding.unbind()
        ```
    """

    def __init__(self, handle: int, semantic: int, dtype: DLDataType, renderer: "Renderer", is_array: bool = False):
        """Initialize attribute binding (internal use - created by Renderer.bind_attribute).

        Args:
            handle: Raw C binding handle value
            semantic: Semantic constant (OVRTX_SEMANTIC_*)
            dtype: Expected DLDataType for tensors used with this binding
            renderer: Renderer instance that created this binding
            is_array: True if this binding is for array attributes
        """
        import weakref

        self._handle = handle
        self._semantic = semantic
        self._dtype = dtype
        self._renderer = weakref.ref(renderer)
        self._is_array = is_array

    def _get_renderer(self) -> "Renderer":
        """Get renderer, raising if it has been destroyed."""
        renderer = self._renderer()
        if renderer is None:
            raise RuntimeError("Renderer has been destroyed")
        return renderer

    @property
    def handle(self) -> int:
        """Raw C binding handle value."""
        return self._handle

    @property
    def semantic(self) -> int:
        """Semantic constant (OVRTX_SEMANTIC_*) for tensor reshaping."""
        return self._semantic

    @property
    def dtype(self) -> DLDataType:
        """Expected DLDataType for tensors used with this binding."""
        return self._dtype

    @property
    def is_array(self) -> bool:
        """True if this binding is for array attributes."""
        return self._is_array

    def write(self, data: _BindingTensorT, dirty_bits: Optional[bytes] = None) -> None:
        """Write attribute data using this binding (synchronous).

        Args:
            data: Data to write. For scalar bindings (bind_attribute): a single DLTensor.
                For array bindings (bind_array_attribute): a list of DLTensor, one per prim.
            dirty_bits: Optional dirty bit array for selective updates.
        """
        tensors = data if self._is_array else [data]
        self._get_renderer()._write_attribute_by_binding(self, tensors, dirty_bits)

    def write_async(
        self, data: _BindingTensorT, dirty_bits: Optional[bytes] = None, data_access: str = "sync"
    ) -> "Operation[None]":
        """Write attribute data using this binding (asynchronous).

        Args:
            data: Data to write. For scalar bindings (bind_attribute): a single DLTensor.
                For array bindings (bind_array_attribute): a list of DLTensor, one per prim.
            dirty_bits: Optional dirty bit array for selective updates.
            data_access: Data access mode ("sync" or "async").

        Returns:
            Operation for async control (yields None on completion).
        """
        tensors = data if self._is_array else [data]
        return self._get_renderer()._write_attribute_by_binding_async(self, tensors, dirty_bits, data_access)

    def map(self, device: str = "cpu", device_id: int = 0) -> "AttributeMapping":
        """Map attribute buffer for direct memory access using this binding.

        Convenience wrapper: delegates to Renderer._map_attribute_by_binding().

        Args:
            device: Device type ("cpu" or "cuda")
            device_id: Device ID (default 0)

        Returns:
            AttributeMapping for direct buffer access
        """
        return self._get_renderer()._map_attribute_by_binding(self, device, device_id)

    def unbind(self) -> None:
        """Release this binding.

        Convenience wrapper: delegates to Renderer._unbind_attribute().
        """
        if self._handle is None:
            return  # Already unbound
        self._get_renderer()._unbind_attribute(self)
        self._handle = None

    def __del__(self):
        """Auto-unbind on garbage collection if not already unbound."""
        if self._handle is None:
            return  # Already unbound
        renderer = self._renderer()  # weakref - may be None if renderer destroyed
        if renderer is None:
            return  # Renderer already gone, can't unbind
        try:
            renderer._unbind_attribute(self)
        except Exception as e:
            print(f"Warning: Exception during AttributeBinding cleanup: {e}", file=sys.stderr)
        self._handle = None

    def __repr__(self) -> str:
        return f"AttributeBinding(handle={self._handle}, semantic={self._semantic})"


class AttributeMapping:
    """High-level wrapper for mapped attribute buffer.

    Provides DLPack-compatible access to RTX-internal buffer for zero-copy writes.
    The buffer is valid until unmap() is called.

    Usage:
        ```python
        # Map attribute - creates handle and buffer
        mapping = renderer.map_attribute(
            prim_paths=["/World/Cube"],
            attribute_name="omni:fabric:worldMatrix",
            dtype=dtype,
            semantic="transform_4x4"
        )

        # Write data via DLPack-compatible framework (zero-copy)
        array = framework.from_dlpack(mapping.tensor)
        array[:] = source_matrix_data

        # Unmap to apply data and destroy mapping
        renderer.unmap_attribute(mapping)
        ```

    Or use as context manager:
        ```python
        with renderer.map_attribute(...) as mapping:
            array = framework.from_dlpack(mapping.tensor)
            array[:] = source_data
        # Automatically unmapped on exit
        ```
    """

    def __init__(
        self,
        mapping: Any,  # ovrtx_attribute_mapping_t from C API
        renderer: "Renderer",
        dltensor: DLTensor,  # Semantic-aware (might be reshaped) DLTensor view (provided by renderer)
        binding_desc: Optional[Any] = None,  # Optional ovrtx_binding_desc_t for write_attribute
        device: str = "cpu",  # Device type for validation ("cpu" or "cuda")
    ):
        """Initialize attribute mapping (internal use - created by Renderer.map_attribute).

        Args:
            mapping: C structure ovrtx_attribute_mapping_t containing map_handle and DLTensor
            renderer: Renderer instance (needed for unmap)
            dltensor: Semantic-aware (might be reshaped) DLTensor view (provided by renderer)
            binding_desc: Optional binding descriptor (for write_attribute calls)
            device: Device type for validation ("cpu" or "cuda")
        """
        self._mapping = mapping
        self._renderer = renderer
        self._dltensor = dltensor
        self._binding_desc = binding_desc
        self._device = device
        self._unmapped = False
        self._managed_tensor: Optional[ManagedDLTensor] = None

    @property
    def tensor(self) -> ManagedDLTensor:
        """Get ManagedDLTensor for DLPack interop (NumPy, Warp, PyTorch, etc.).

        Returns a semantic-aware view optimized for NumPy users:
        - Matrix4d (TRANSFORM_4x4): Reshaped to (4, 4) for direct matrix operations
        - RGBA images (COLOR_RGBA4b): Shape (H, W, 4) - standard image format
        - RGB images (COLOR_RGB3f): Shape (H, W, 3) - standard image format
        - Other types: Standard DLPack expansion

        Returns:
            ManagedDLTensor with semantic-aware shape optimization

        Raises:
            RuntimeError: If accessed after unmap().
        """
        if self._unmapped:
            raise RuntimeError("Mapping already unmapped - tensor no longer valid")
        if self._managed_tensor is None:
            # Mapped attributes are writeable (user writes data via this tensor)
            self._managed_tensor = ManagedDLTensor(self._dltensor, self, None, readonly=False)
        return self._managed_tensor

    @property
    def map_handle(self) -> int:
        """Get the map handle for unmap operation.

        Returns:
            Map handle value (uint64)
        """
        return int(self._mapping.map_handle)

    @property
    def binding_desc(self) -> Optional[Any]:
        """Get the binding descriptor used for map_attribute (if available).

        This can be used for write_attribute calls if needed.

        Returns:
            ovrtx_binding_desc_t or None
        """
        return self._binding_desc

    @property
    def device(self) -> str:
        """Get the device type this attribute is mapped to.

        Returns:
            Device type string ("cpu" or "cuda")
        """
        return self._device

    def unmap(self, event: Optional[int] = None, stream: Optional[int] = None) -> None:
        """Unmap attribute buffer with optional CUDA synchronization.

        For CUDA-mapped attributes, you can provide a CUDA event or stream handle
        to synchronize with ongoing GPU work before the data is consumed.

        Args:
            event: CUDA event handle to wait on before accessing data (from wp.Event.cuda_event).
                   The C library will wait for this event before reading the mapped data.
            stream: CUDA stream handle for synchronization (from wp.Stream.cuda_stream).
                    The C library will synchronize on this stream before reading the data.

        Raises:
            ValueError: If event/stream provided for CPU-mapped attribute.
            ValueError: If both event and stream provided (mutually exclusive).

        Note:
            If called within a context manager, __exit__ becomes a no-op (already unmapped).
            If not called, context manager's __exit__ unmaps without CUDA sync (fallback).

        Example:
            ```python
            with renderer.map_attribute(..., device="cuda") as mapping:
                stream = wp.Stream()
                wp_array = wp.from_dlpack(mapping.tensor)
                wp.launch(kernel=my_kernel, inputs=[wp_array], stream=stream)
                # Unmap with stream sync - C library waits for stream before consuming data
                mapping.unmap(stream=stream.cuda_stream)
            ```
        """
        if self._unmapped:
            return
        self._managed_tensor = None
        self._renderer.unmap_attribute(self, event=event, stream=stream)

    def __enter__(self) -> "AttributeMapping":
        """Context manager entry - returns self."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - automatically unmap if not already unmapped."""
        self.unmap()
        return False

    def __repr__(self) -> str:
        status = "unmapped" if self._unmapped else "mapped"
        return f"AttributeMapping(map_handle={self.map_handle}, {status})"


@dataclass
class RendererConfig:
    "ovrtx configuration with automatic defaults."

    #: Enables synchronous rendering mode for debugging purposes. 
    sync_mode: Optional[bool] = None
    #: Set the path to the log file for logging output.
    log_file_path: Optional[str] = None
    #: Set the log level for logging output: "verbose", "info", "warn", "error"
    log_level: Optional[str] = None

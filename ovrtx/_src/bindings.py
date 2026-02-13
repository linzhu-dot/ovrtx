# Copyright (c) 2025-2026, NVIDIA CORPORATION. All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

import ctypes
import os
import sys
from pathlib import Path
from typing import Any, List, Optional

from .dlpack import DLDataType, DLTensor


class ovx_string_t(ctypes.Structure):
    """ovx_string_t binding class"""

    _fields_ = [  # these must match the ovx_string_t struct in ovx_types.h
        ("ptr", ctypes.c_char_p),
        ("length", ctypes.c_size_t),
    ]

    def __init__(self, value=None):
        """Constructor that accepts any value convertible to string.

        Args:
            value: Any value that can be converted to string using str().
                  If None, creates an empty string structure.
        """
        super().__init__()

        encoded = str(value if value is not None else "").encode("utf-8")
        self._bytes = ctypes.create_string_buffer(encoded)
        self.ptr = ctypes.cast(self._bytes, ctypes.c_char_p)
        self.length = len(encoded)

    def __bool__(self) -> bool:
        """Return True if string is non-null and non-empty."""
        return self.ptr is not None and self.length > 0

    def __len__(self) -> int:
        """Return byte length (excluding null terminator)."""
        return int(self.length)

    def __str__(self) -> str:
        """Convert to Python string. Treats string as null-terminated."""
        return ctypes.string_at(self.ptr).decode("utf-8", errors="replace") if self.ptr is not None else ""

    def __repr__(self):
        value_repr = repr(str(self)) if self.ptr is not None else "None"
        return f"ovx_string_t({value_repr}, ptr={self.ptr}, length={self.length})"


OVRTX_CONFIG_VALUE_BOOL: int = 0
OVRTX_CONFIG_VALUE_INT64: int = 1
OVRTX_CONFIG_VALUE_UINT64: int = 2
OVRTX_CONFIG_VALUE_DOUBLE: int = 3
OVRTX_CONFIG_VALUE_STRING: int = 4
OVRTX_CONFIG_VALUE_BLOB: int = 5


class ovrtx_renderer_config_value_type_t(ctypes.c_int):
    """Value type discriminator."""

    pass


class ovrtx_config_blob_t(ctypes.Structure):
    """Blob data structure."""

    _fields_ = [
        ("data", ctypes.c_void_p),
        ("size", ctypes.c_size_t),
    ]


class ovrtx_renderer_config_value_t(ctypes.Structure):
    """Tagged union value for config entries."""

    class _ValueUnion(ctypes.Union):
        _fields_ = [
            ("bool_value", ctypes.c_bool),
            ("int_value", ctypes.c_int64),
            ("uint_value", ctypes.c_uint64),
            ("double_value", ctypes.c_double),
            ("string_value", ovx_string_t),
            ("blob_value", ovrtx_config_blob_t),
        ]

    _fields_ = [
        ("type", ovrtx_renderer_config_value_type_t),
        ("_union", _ValueUnion),
    ]


class ovrtx_renderer_config_entry_t(ctypes.Structure):
    """Config entry with self-contained lifetime management."""

    _fields_ = [
        ("key", ovx_string_t),
        ("value", ovrtx_renderer_config_value_t),
    ]

    def __init__(self, key: str, value: ovrtx_renderer_config_value_t, data: Optional[Any] = None):
        """Create entry, retaining ownership of key and value."""
        self._key = ovx_string_t(key)
        if data is not None:
            self._data = data

        super().__init__(key=self._key, value=value)

    @classmethod
    def from_bool(cls, key: str, value: bool) -> "ovrtx_renderer_config_entry_t":
        value_union = ovrtx_renderer_config_value_t._ValueUnion(bool_value=value)
        return cls(key, ovrtx_renderer_config_value_t(OVRTX_CONFIG_VALUE_BOOL, value_union))

    @classmethod
    def from_int64(cls, key: str, value: int) -> "ovrtx_renderer_config_entry_t":
        value_union = ovrtx_renderer_config_value_t._ValueUnion(int_value=value)
        return cls(key, ovrtx_renderer_config_value_t(OVRTX_CONFIG_VALUE_INT64, value_union))

    @classmethod
    def from_uint64(cls, key: str, value: int) -> "ovrtx_renderer_config_entry_t":
        value_union = ovrtx_renderer_config_value_t._ValueUnion(uint_value=value)
        return cls(key, ovrtx_renderer_config_value_t(OVRTX_CONFIG_VALUE_UINT64, value_union))

    @classmethod
    def from_double(cls, key: str, value: float) -> "ovrtx_renderer_config_entry_t":
        value_union = ovrtx_renderer_config_value_t._ValueUnion(double_value=value)
        return cls(key, ovrtx_renderer_config_value_t(OVRTX_CONFIG_VALUE_DOUBLE, value_union))

    @classmethod
    def from_bytes(cls, key: str, data: bytes) -> "ovrtx_renderer_config_entry_t":
        if not data:
            value_union = ovrtx_renderer_config_value_t._ValueUnion(blob_value=ovrtx_config_blob_t(data=None, size=0))
            return cls(key, ovrtx_renderer_config_value_t(OVRTX_CONFIG_VALUE_BLOB, value_union))
        else:
            # create a new ctypes bytes buffer from the data ...
            buffer = (ctypes.c_ubyte * len(data)).from_buffer_copy(data)
            blob = ovrtx_config_blob_t(data=ctypes.cast(buffer, ctypes.c_void_p), size=len(buffer))
            # ... and pass it to the constructor, which will keep it alive through its data argument
            # (otherwise only the ctypes union fields will be copied during the assignment)
            value_union = ovrtx_renderer_config_value_t._ValueUnion(blob_value=blob)
            return cls(key, ovrtx_renderer_config_value_t(OVRTX_CONFIG_VALUE_BLOB, value_union), buffer)

    @classmethod
    def from_string(cls, key: str, value: str) -> "ovrtx_renderer_config_entry_t":
        # create a new ovx_string_t object and keep it alive through the constructor's data argument
        # (otherwise only the ctypes struct fields will be copied during the assignment)
        string_value = ovx_string_t(value)
        value_union = ovrtx_renderer_config_value_t._ValueUnion(string_value=string_value)
        return cls(key, ovrtx_renderer_config_value_t(OVRTX_CONFIG_VALUE_STRING, value_union), string_value)


class ovrtx_config_t(ctypes.Structure):
    """Config container with self-contained lifetime management."""

    _fields_ = [
        ("entries", ctypes.POINTER(ovrtx_renderer_config_entry_t)),
        ("entry_count", ctypes.c_size_t),
    ]

    def __init__(self, entries: list[ovrtx_renderer_config_entry_t]):
        """Create config, retaining ownership of entries array."""
        if entries:
            # Defensively keep array AND individual entries alive to prevent garbage collecting pointer-wrapped objects
            self._array = (ovrtx_renderer_config_entry_t * len(entries))(*entries)
            self._entries = entries
            super().__init__(entries=self._array, entry_count=len(entries))
        else:
            super().__init__(entries=None, entry_count=0)
            self._array = None
            self._entries = []


class ovrtx_renderer_t(ctypes.Structure):
    """ovrtx_renderer_t binding class (fully opaque)"""

    pass


# OVRTX API Status Constants
OVRTX_API_SUCCESS: int = 0
OVRTX_API_ERROR: int = 1
OVRTX_API_TIMEOUT: int = 2


class ovrtx_api_status_t(ctypes.c_int32):
    """ovrtx_api_status_t binding class"""

    def __eq__(self, other):
        return self.value == int(other) if isinstance(other, (int, ovrtx_api_status_t)) else NotImplemented

    def __ne__(self, other):
        result = self.__eq__(other)
        return not result if result is not NotImplemented else NotImplemented


class ovrtx_result_t(ctypes.Structure):
    """ovrtx_result_t binding class."""

    _fields_ = [
        ("status", ovrtx_api_status_t),
    ]


class ovrtx_timeout_t(ctypes.Structure):
    """ovrtx_timeout_t binding class"""

    _fields_ = [
        ("time_out_ns", ctypes.c_uint64),
    ]


OVRTX_TIMEOUT_INFINITE = ovrtx_timeout_t(time_out_ns=0xFFFFFFFFFFFFFFFF)


class ovrtx_op_id_t(ctypes.c_uint64):
    """ovrtx_op_id_t binding class for operation IDs"""

    pass


class ovrtx_op_wait_result_t(ctypes.Structure):
    """ovrtx_op_wait_result_t binding class.

    Note: error_op_ids contains op IDs that errored. Use get_last_op_error(op_id)
    to get the error string for each failed operation.
    """

    _fields_ = [
        ("error_op_ids", ctypes.POINTER(ovrtx_op_id_t)),
        ("num_error_ops", ctypes.c_size_t),
        ("lowest_pending_op_id", ovrtx_op_id_t),
    ]


class ovrtx_usd_input_t(ctypes.Structure):
    """USD input descriptor - supports file path, stage ID, or layer content.

    Exactly one field should be set:
    - usd_file_path: Path to USD file (most common, stable)
    - usd_stage_id: ID of existing USD runtime stage (experimental)
    - usd_layer_content: Inline USDA content (experimental)
    """

    _fields_ = [
        ("usd_file_path", ovx_string_t),  # empty/None to disable
        ("usd_stage_id", ctypes.c_uint64),  # 0 to disable
        ("usd_layer_content", ovx_string_t),  # empty/None to disable
    ]


class ovrtx_usd_handle_t(ctypes.c_uint64):
    """ovrtx_usd_handle_t binding class for USD file handles"""

    pass


class ovrtx_enqueue_result_t(ctypes.Structure):
    """ovrtx_enqueue_result_t binding class."""

    _fields_ = [
        ("status", ovrtx_api_status_t),
        ("op_index", ovrtx_op_id_t),
    ]


class ovrtx_render_product_set_t(ctypes.Structure):
    """ovrtx_render_product_set_t binding class"""

    _fields_ = [
        ("render_products", ctypes.POINTER(ovx_string_t)),
        ("num_render_products", ctypes.c_size_t),
    ]


class ovrtx_step_result_handle_t(ctypes.c_uint64):
    """ovrtx_step_result_handle_t binding class for step result handles"""

    pass


# Event status enum (for fetch_results)
OVRTX_EVENT_PENDING = 0
OVRTX_EVENT_COMPLETED = 1
OVRTX_EVENT_FAILURE = 2

# Map device type enum (for map_rendered_output)
OVRTX_MAP_DEVICE_TYPE_DEFAULT = 0  # Most efficient format
OVRTX_MAP_DEVICE_TYPE_CPU = 1  # CPU memory (synchronous copy from GPU)
OVRTX_MAP_DEVICE_TYPE_CUDA = 2  # Raw CUDA device memory (may incur copy for images)
OVRTX_MAP_DEVICE_TYPE_CUDA_ARRAY = 3  # CUDA array as kDLOpaqueHandle (saves copy for images)


# Simple type alias - rendered output handles are just uint64 values
# We don't derive from ctypes types to avoid instance creation issues
ovrtx_rendered_output_handle_t = ctypes.c_uint64


# Render output metadata structures
class ovrtx_render_product_render_var_output_t(ctypes.Structure):
    """Single render variable output (e.g., 'rgb', 'depth')."""

    _fields_ = [
        ("render_var_name", ovx_string_t),
        ("output_handle", ctypes.c_uint64),  # Plain c_uint64, not type alias
    ]


class ovrtx_render_product_frame_output_t(ctypes.Structure):
    """Single frame output with multiple render variables."""

    _fields_ = [
        ("frame_start_time", ctypes.c_double),
        ("frame_end_time", ctypes.c_double),
        ("output_render_vars", ctypes.POINTER(ovrtx_render_product_render_var_output_t)),
        ("render_var_count", ctypes.c_size_t),
    ]


class ovrtx_render_product_output_t(ctypes.Structure):
    """Single render product output with multiple frames."""

    _fields_ = [
        ("render_product_path", ovx_string_t),
        ("output_frames_produced", ctypes.c_float),
        ("output_frames", ctypes.POINTER(ovrtx_render_product_frame_output_t)),
        ("output_frame_count", ctypes.c_size_t),
    ]


class ovrtx_render_product_set_outputs_t(ctypes.Structure):
    """Complete output metadata from a step operation."""

    _fields_ = [
        ("status", ctypes.c_int),
        ("error_message", ovx_string_t),
        ("simulation_start_time", ctypes.c_double),
        ("simulation_end_time", ctypes.c_double),
        ("outputs", ctypes.POINTER(ovrtx_render_product_output_t)),
        ("output_count", ctypes.c_size_t),
        ("start_time", ctypes.c_double),
        ("end_time", ctypes.c_double),
    ]


class ovrtx_cuda_sync_t(ctypes.Structure):
    """CUDA synchronization hints.

    Matches ovrtx_cuda_sync_t from ovrtx_types.h.

    Fields:
        stream: CUDA stream to synchronize to. 0 = no sync, 1 = default stream, >1 = specific stream
        wait_event: Event to wait on before operation (0 = none)
    """

    _fields_ = [
        ("stream", ctypes.c_size_t),  # uintptr_t
        ("wait_event", ctypes.c_size_t),  # uintptr_t
    ]


class ovrtx_output_buffer_t(ctypes.Structure):
    """Output buffer wrapping a DLTensor with optional CUDA synchronization.

    This is an ovrtx-specific extension wrapping the standard DLTensor.
    """

    _fields_ = [
        ("dl", DLTensor),  # DLPack tensor
        ("cuda_sync", ovrtx_cuda_sync_t),  # CUDA sync info (stream + event)
    ]


# Type alias for map handles (uint64)
ovrtx_rendered_output_map_handle_t = ctypes.c_uint64


class ovrtx_map_output_description_t(ctypes.Structure):
    """Description for mapping rendered output to specific device.

    Fields:
        device_type: ovrtx_map_device_type_t (DEFAULT=0, CPU=1, CUDA=2, CUDA_ARRAY=3)
        sync_stream: uintptr_t for CUDA stream (0=no sync, 1=default stream, >1=specific stream)
    """

    _fields_ = [
        ("device_type", ctypes.c_int),
        ("sync_stream", ctypes.c_void_p),
    ]


class ovrtx_rendered_output_t(ctypes.Structure):
    """Rendered output with buffer and metadata from map operation."""

    _fields_ = [
        ("status", ctypes.c_int),  # ovrtx_event_status_t
        ("error_message", ovx_string_t),
        ("map_handle", ctypes.c_uint64),  # ovrtx_rendered_output_map_handle_t
        ("name", ovx_string_t),
        ("buffer", ovrtx_output_buffer_t),
    ]


# ============================================================================
# Phase 1.1: Attribute write structures and enums
# ============================================================================

# Enum constants for ovrtx_attribute_semantic_t
OVRTX_SEMANTIC_NONE = 0
OVRTX_SEMANTIC_TRANSFORM_4x4 = 1  # Column-major 4x4 double matrix (kDLFloat, 64, 16)
OVRTX_SEMANTIC_TRANSFORM_POS3d_ROT4f_SCALE3f = 2  # 3xdouble pos + 4xfloat rot + 3xfloat scale (kDLUint, 8, 52)
OVRTX_SEMANTIC_TRANSFORM_POS3d_ROT3x3f = 3  # 3xdouble pos + 3x3float rotation matrix (kDLUint, 8, 60)
OVRTX_SEMANTIC_PATH_STRING = 4  # Prim paths as ovx_string_t (kDLUint, 128, 1). Sync data access only.
OVRTX_SEMANTIC_TOKEN_STRING = 5  # String tokens as ovx_string_t (kDLUint, 128, 1). Sync data access only.
OVRTX_SEMANTIC_COLOR_RGBA4b = 6  # Color as 4 bytes (kDLUint, 8, 4)
OVRTX_SEMANTIC_COLOR_RGB3f = 7  # Color as 3 floats (kDLFloat, 32, 3)


class ovrtx_attribute_semantic_t(ctypes.c_int):
    """Attribute semantic type enum."""

    pass


# Enum constants for ovrtx_binding_prim_mode_t
OVRTX_BINDING_PRIM_MODE_EXISTING_ONLY = 0
OVRTX_BINDING_PRIM_MODE_MUST_EXIST = 1
OVRTX_BINDING_PRIM_MODE_CREATE_NEW = 2


class ovrtx_binding_prim_mode_t(ctypes.c_int):
    """Prim binding mode enum."""

    pass


# Enum constants for ovrtx_binding_flag_t
OVRTX_BINDING_FLAG_NONE = 0
OVRTX_BINDING_FLAG_OPTIMIZE = 1 << 0


class ovrtx_binding_flag_t(ctypes.c_int):
    """Binding flag enum."""

    pass


# Enum constants for ovrtx_data_access_t
OVRTX_DATA_ACCESS_ASYNC = 0
OVRTX_DATA_ACCESS_SYNC = 1


class ovrtx_data_access_t(ctypes.c_int):
    """Data access mode enum."""

    pass


# Type aliases
ovrtx_attribute_binding_handle_t = ctypes.c_uint64
ovrtx_map_handle_t = ctypes.c_uint64


class ovx_string_or_token_t(ctypes.Structure):
    """String or token union for attribute names."""

    _fields_ = [
        ("token", ctypes.c_uint64),  # ovx_token_t
        ("string", ovx_string_t),
    ]


class ovrtx_prim_list_t(ctypes.Structure):
    """List of prim paths."""

    _fields_ = [
        ("prim_paths", ctypes.POINTER(ovx_string_t)),  # const ovx_string_t*
        ("num_paths", ctypes.c_size_t),
    ]


class ovrtx_attribute_type_t(ctypes.Structure):
    """Attribute type descriptor."""

    _fields_ = [
        ("dtype", DLDataType),
        ("is_array", ctypes.c_bool),
        ("semantic", ovrtx_attribute_semantic_t),
    ]


class ovrtx_binding_desc_t(ctypes.Structure):
    """Attribute binding descriptor."""

    _fields_ = [
        ("prim_list", ovrtx_prim_list_t),
        ("prims_list_handle", ctypes.c_uint64),  # ovx_primpath_list_t
        ("attribute_name", ovx_string_or_token_t),
        ("attribute_type", ovrtx_attribute_type_t),
        ("prim_mode", ovrtx_binding_prim_mode_t),
        ("flags", ovrtx_binding_flag_t),
    ]


class ovrtx_binding_desc_or_handle_t(ctypes.Structure):
    """Binding descriptor or handle union."""

    _fields_ = [
        ("binding_desc", ovrtx_binding_desc_t),
        ("binding_handle", ovrtx_attribute_binding_handle_t),
    ]


class ovrtx_input_buffer_t(ctypes.Structure):
    """Input buffer for attribute writes."""

    _fields_ = [
        ("tensors", ctypes.POINTER(DLTensor)),
        ("tensor_count", ctypes.c_uint64),
        ("dirty_bits", ctypes.POINTER(ctypes.c_uint8)),
        ("dirty_bits_size", ctypes.c_size_t),
        ("access_cuda_sync", ovrtx_cuda_sync_t),
        ("done_cuda_sync", ovrtx_cuda_sync_t),
    ]


class ovrtx_mapping_desc_t(ctypes.Structure):
    """Descriptor for attribute mapping."""

    _fields_ = [
        ("device_type", ctypes.c_int32),  # DLDevice device_type
        ("device_id", ctypes.c_int32),  # DLDevice device_id
    ]


class ovrtx_attribute_mapping_t(ctypes.Structure):
    """Result from map_attribute."""

    _fields_ = [
        ("map_handle", ovrtx_map_handle_t),  # Handle for unmap operation
        ("dl", DLTensor),  # Mapped memory as tensor, valid until unmap
    ]


class Bindings:
    """Wrapper for configured C library functions.

    Error Handling:
        Methods return ovrtx_result_t with a status field. On failure (status != OVRTX_API_SUCCESS),
        call get_last_error() to retrieve the error message. The error string is thread-local and
        valid until the next API call on the same thread.
    """

    def __init__(self, lib: ctypes.CDLL):
        self._lib: ctypes.CDLL = lib

    @property
    def library_path(self) -> Path:
        """Get the path of the loaded library."""
        return Path(self._lib._name)

    def create_renderer(self, config: ovrtx_config_t) -> tuple[ovrtx_result_t, Any]:
        """Create a new renderer instance.

        Args:
            config: Renderer configuration structure.

        Returns:
            Tuple of (result, renderer_handle). Handle is None if creation failed.
        """
        renderer_handle = ctypes.POINTER(ovrtx_renderer_t)()
        result = self._lib.ovrtx_create_renderer(ctypes.byref(config), ctypes.byref(renderer_handle))
        handle = renderer_handle if result.status == OVRTX_API_SUCCESS else None
        return result, handle

    def destroy_renderer(self, renderer_handle: Any) -> ovrtx_result_t:
        """Destroy renderer and release resources.

        Args:
            renderer_handle: Renderer handle from create_renderer, or None.

        Returns:
            Result with status.
        """
        return self._lib.ovrtx_destroy_renderer(renderer_handle)

    def get_last_error(self) -> str:
        """Get the error string from the last failed API call on this thread.

        The error string is valid until the next API call (or any call that might
        trigger a fiber switch) on the same thread.

        Returns:
            Error message string, or empty string if no error.
        """
        error = self._lib.ovrtx_get_last_error()
        return str(error) if error.ptr else ""

    def get_last_op_error(self, op_id: ovrtx_op_id_t) -> str:
        """Get the error string for a specific failed operation.

        The error string is valid until the next call to wait_op (or any call
        that might trigger a fiber switch) on the same thread.

        Args:
            op_id: Operation ID to get error for.

        Returns:
            Error message string, or empty string if no error.
        """
        error = self._lib.ovrtx_get_last_op_error(op_id)
        return str(error) if error.ptr else ""

    def wait_op(
        self, renderer_handle: Any, op_id: ovrtx_op_id_t, timeout: ovrtx_timeout_t
    ) -> tuple[ovrtx_result_t, ovrtx_op_wait_result_t]:
        """Wait for an operation to complete.

        Args:
            renderer_handle: Renderer handle from create_renderer.
            op_id: Operation ID to wait for.
            timeout: Timeout for the operation.

        Returns:
            Tuple of (result, wait_result).
        """
        c_wait_result = ovrtx_op_wait_result_t()
        result = self._lib.ovrtx_wait_op(renderer_handle, op_id, timeout, ctypes.byref(c_wait_result))
        return result, c_wait_result

    def add_usd(
        self, renderer_handle: Any, usd_input: ovrtx_usd_input_t, path_prefix: Optional[str] = None
    ) -> tuple[ovrtx_enqueue_result_t, Any]:
        """Add USD content to the renderer.

        Args:
            renderer_handle: Renderer handle from create_renderer.
            usd_input: USD input descriptor (prepared by caller).
            path_prefix: Optional path prefix for the USD stage.

        Returns:
            Tuple of (enqueue_result, usd_handle). usd_handle is None if addition failed.
        """
        usd_handle = ovrtx_usd_handle_t()
        result = self._lib.ovrtx_add_usd(
            renderer_handle, usd_input, ovx_string_t(path_prefix), ctypes.byref(usd_handle)
        )

        handle = usd_handle if result.status == OVRTX_API_SUCCESS else None
        return result, handle

    def clone_usd(
        self, renderer_handle: Any, source_path: ovx_string_t, target_paths_array: Any, num_targets: int
    ) -> ovrtx_enqueue_result_t:
        """Clone a USD subtree to one or more target paths.

        Args:
            renderer_handle: Renderer handle from create_renderer.
            source_path: Path to the source prim to clone.
            target_paths_array: ctypes array of ovx_string_t target paths.
            num_targets: Number of target paths.

        Returns:
            Enqueue result with operation status and op_index.
        """
        return self._lib.ovrtx_clone_usd(renderer_handle, source_path, target_paths_array, num_targets)

    def remove_usd(self, renderer_handle: Any, usd_handle: ovrtx_usd_handle_t) -> ovrtx_enqueue_result_t:
        """Remove a previously added USD file from the stage.

        Args:
            renderer_handle: Renderer handle from create_renderer.
            usd_handle: Handle returned by add_usd.

        Returns:
            Enqueue result with operation status and op_index.
        """
        return self._lib.ovrtx_remove_usd(renderer_handle, usd_handle)

    def update_stage_from_usd_time(self, renderer_handle: Any, usd_time: float) -> ovrtx_enqueue_result_t:
        """Update stage attributes from USD time samples.

        Args:
            renderer_handle: Renderer handle from create_renderer.
            usd_time: USD time code to evaluate.

        Returns:
            Enqueue result with operation status and op_index.
        """
        return self._lib.ovrtx_update_stage_from_usd_time(renderer_handle, usd_time)

    def reset_stage(self, renderer_handle: Any) -> ovrtx_enqueue_result_t:
        """Reset the runtime stage to empty.

        Args:
            renderer_handle: Renderer handle from create_renderer.

        Returns:
            Enqueue result with operation status and op_index.
        """
        return self._lib.ovrtx_reset_stage(renderer_handle)

    def step(
        self, renderer_handle: Any, render_product_set: ovrtx_render_product_set_t, delta_time: float
    ) -> tuple[ovrtx_enqueue_result_t, Any]:
        """Enqueue a step operation.

        Args:
            renderer_handle: Renderer handle from create_renderer.
            render_product_set: Render product set to step.
            delta_time: Time step for simulation.

        Returns:
            Tuple of (enqueue_result, step_result_handle). Handle is None if step failed.
        """
        step_result_handle = ovrtx_step_result_handle_t()
        result = self._lib.ovrtx_step(renderer_handle, render_product_set, delta_time, ctypes.byref(step_result_handle))
        handle = step_result_handle if result.status == OVRTX_API_SUCCESS else None
        return result, handle

    def fetch_results(
        self,
        renderer: Any,
        step_result_handle: ovrtx_step_result_handle_t,
        timeout: ovrtx_timeout_t,
    ) -> tuple[ovrtx_result_t, ovrtx_render_product_set_outputs_t]:
        """Fetch rendering results metadata.

        Args:
            renderer: Renderer instance pointer
            step_result_handle: Handle from step operation (ovrtx_step_result_handle_t)
            timeout: Timeout configuration (ovrtx_timeout_t)

        Returns:
            Tuple of (result, output_metadata)
        """
        output_metadata = ovrtx_render_product_set_outputs_t()
        result = self._lib.ovrtx_fetch_results(renderer, step_result_handle, timeout, ctypes.byref(output_metadata))
        return result, output_metadata

    def destroy_results(
        self,
        renderer: Any,
        step_result_handle: ovrtx_step_result_handle_t,
    ) -> ovrtx_result_t:
        """Destroy step results and free associated resources.

        Args:
            renderer: Renderer instance pointer
            step_result_handle: Step result handle to destroy

        Returns:
            Result status
        """
        return self._lib.ovrtx_destroy_results(renderer, step_result_handle)

    def map_rendered_output(
        self,
        renderer: Any,
        output_handle: ovrtx_rendered_output_handle_t,
        map_output_description: Any,
        timeout: ovrtx_timeout_t,
    ) -> tuple[ovrtx_result_t, ovrtx_rendered_output_t]:
        rendered_output = ovrtx_rendered_output_t()
        result = self._lib.ovrtx_map_rendered_output(
            renderer,
            output_handle,
            map_output_description,
            timeout,
            ctypes.byref(rendered_output),
        )
        return result, rendered_output

    def unmap_rendered_output(
        self,
        renderer: Any,
        map_handle: ovrtx_rendered_output_map_handle_t,
        before_destroy_cuda_sync: Optional[ovrtx_cuda_sync_t] = None,
    ) -> ovrtx_result_t:
        """Unmap rendered output (low-level).

        Args:
            renderer: Renderer instance pointer
            map_handle: Map handle from rendered output (ovrtx_rendered_output_map_handle_t)
            before_destroy_cuda_sync: Optional CUDA sync to wait for before destroying/reusing
                the mapped memory. Pass None for no sync (default).

        Returns:
            Result status (ovrtx_result_t)
        """
        sync = before_destroy_cuda_sync if before_destroy_cuda_sync is not None else ovrtx_cuda_sync_t()
        return self._lib.ovrtx_unmap_rendered_output(renderer, map_handle, sync)

    def write_attribute(
        self,
        renderer: Any,
        binding_handle_or_desc: ovrtx_binding_desc_or_handle_t,
        data_array: ovrtx_input_buffer_t,
        data_access: ovrtx_data_access_t,
    ) -> ovrtx_enqueue_result_t:
        """Write attribute data (low-level).

        Args:
            renderer: Renderer instance pointer
            binding_handle_or_desc: Binding descriptor or handle
            data_array: Input buffer containing DLTensor array
            data_access: Data access mode (SYNC or ASYNC)

        Returns:
            Enqueue result with operation ID
        """
        return self._lib.ovrtx_write_attribute(
            renderer, ctypes.byref(binding_handle_or_desc), ctypes.byref(data_array), data_access
        )

    def create_attribute_binding(
        self,
        renderer: Any,
        description: ovrtx_binding_desc_t,
    ) -> tuple[ovrtx_enqueue_result_t, ovrtx_attribute_binding_handle_t]:
        """Create a persistent attribute binding handle.

        Args:
            renderer: Renderer handle
            description: Binding description

        Returns:
            Tuple of (enqueue result, binding handle)
        """
        handle = ovrtx_attribute_binding_handle_t()
        result = self._lib.ovrtx_create_attribute_binding(renderer, ctypes.byref(description), ctypes.byref(handle))
        return result, handle

    def destroy_attribute_binding(
        self,
        renderer: Any,
        binding_handle: ovrtx_attribute_binding_handle_t,
    ) -> ovrtx_enqueue_result_t:
        """Destroy a persistent attribute binding handle.

        Args:
            renderer: Renderer handle
            binding_handle: Binding handle to destroy

        Returns:
            Enqueue result with operation ID
        """
        return self._lib.ovrtx_destroy_attribute_binding(renderer, binding_handle)

    def map_attribute(
        self,
        renderer: Any,
        binding_handle_or_desc: ovrtx_binding_desc_or_handle_t,
        mapping_desc: ovrtx_mapping_desc_t,
    ) -> tuple[ovrtx_result_t, ovrtx_attribute_mapping_t]:
        """Map attribute to get direct access to RTX-internal buffer.

        Args:
            renderer: Renderer handle
            binding_handle_or_desc: Binding descriptor or handle
            mapping_desc: Mapping descriptor (device type and ID)

        Returns:
            Tuple of (result, attribute_mapping). Mapping contains map_handle and DLTensor.
        """
        mapping = ovrtx_attribute_mapping_t()
        result = self._lib.ovrtx_map_attribute(
            renderer, ctypes.byref(binding_handle_or_desc), mapping_desc, ctypes.byref(mapping)
        )
        return result, mapping

    def unmap_attribute(
        self,
        renderer: Any,
        map_handle: ovrtx_map_handle_t,
        cuda_sync: ovrtx_cuda_sync_t,
    ) -> ovrtx_enqueue_result_t:
        """Unmap attribute and apply written data to stage representation.

        Args:
            renderer: Renderer handle
            map_handle: Map handle from map_attribute
            cuda_sync: CUDA synchronization hints (optional, can be empty)

        Returns:
            Enqueue result with operation ID
        """
        return self._lib.ovrtx_unmap_attribute(renderer, map_handle, cuda_sync)

    def reset(self, renderer_handle: Any, time: float) -> ovrtx_enqueue_result_t:
        """Reset sensor simulation history to a specific time.

        Args:
            renderer_handle: Renderer handle from create_renderer.
            time: Simulation time to reset to.

        Returns:
            Enqueue result with operation status and op_index.
        """
        return self._lib.ovrtx_reset(renderer_handle, time)


OVRTX_LIBRARY_PATH_HINT: Optional[str] = None


class _LibraryLoader:
    """Internal class to lazily load a singleton ovrtx library and create bindings for it."""

    def __init__(self):
        self._lib: Optional[ctypes.CDLL] = None
        self._lib_name: str = "ovrtx-dynamic.dll" if sys.platform.startswith("win") else "libovrtx-dynamic.so"
        self._lib_search_paths: List[Path] = []

        # Populate standard library search paths by precedence:
        # 1. <ovrtx_package_dir>/bin
        # 2. Current working directory
        # 3. all paths in LD_LIBRARY_PATH (Linux only)
        # 4. all paths in PATH
        search_paths = [Path(__file__).parent.parent / "bin", Path.cwd()]  # Go up from _src to ovrtx, then bin

        if not sys.platform.startswith("win"):
            if ld_paths := os.environ.get("LD_LIBRARY_PATH", ""):
                search_paths.extend([Path(p) for p in ld_paths.split(os.pathsep) if p])

        if path_paths := os.environ.get("PATH", ""):
            search_paths.extend([Path(p) for p in path_paths.split(os.pathsep) if p])

        # Resolve and filter valid directories
        self._lib_search_paths = [p.resolve() for p in search_paths if p.exists() and p.is_dir()]

    def _cleanup(self):
        """Cleanup function called at interpreter exit."""
        if self._lib is not None:
            try:
                result = self._lib.ovrtx_shutdown()
                if result.status != OVRTX_API_SUCCESS:
                    error = self._lib.ovrtx_get_last_error()
                    error_msg = str(error) if error.ptr else "Unknown error"
                    print(f"Warning: Failed to shutdown ovrtx library: {error_msg}", file=sys.stderr)
            except Exception as e:
                # During shutdown, modules may be partially torn down
                print(f"Warning: Exception during ovrtx library cleanup: {e}", file=sys.stderr)
            finally:
                self._lib = None

    def create_bindings(self, config: ovrtx_config_t) -> Bindings:
        """Get bindings wrapper, loading library lazily on first access.

        Args:
            config: Configuration for library/renderer initialization.

        Returns:
            Bindings wrapper with configured function signatures.

        Raises:
            RuntimeError: If library cannot be found or functions cannot be bound.
        """
        self._check_pxr_not_available()

        if self._lib is None:
            lib = None  # Initialize for exception handler
            try:
                library_path_hint = [Path(OVRTX_LIBRARY_PATH_HINT)] if OVRTX_LIBRARY_PATH_HINT else None
                lib = self._load_library(library_path_hint)

                # Configure function signatures
                lib.ovrtx_initialize.argtypes = [ctypes.POINTER(ovrtx_config_t)]
                lib.ovrtx_initialize.restype = ovrtx_result_t

                lib.ovrtx_shutdown.argtypes = []
                lib.ovrtx_shutdown.restype = ovrtx_result_t

                lib.ovrtx_create_renderer.argtypes = [
                    ctypes.POINTER(ovrtx_config_t),
                    ctypes.POINTER(ctypes.POINTER(ovrtx_renderer_t)),
                ]
                lib.ovrtx_create_renderer.restype = ovrtx_result_t

                lib.ovrtx_destroy_renderer.argtypes = [ctypes.POINTER(ovrtx_renderer_t)]
                lib.ovrtx_destroy_renderer.restype = ovrtx_result_t

                lib.ovrtx_get_last_error.argtypes = []
                lib.ovrtx_get_last_error.restype = ovx_string_t

                lib.ovrtx_get_last_op_error.argtypes = [ovrtx_op_id_t]
                lib.ovrtx_get_last_op_error.restype = ovx_string_t

                lib.ovrtx_wait_op.argtypes = [
                    ctypes.POINTER(ovrtx_renderer_t),
                    ovrtx_op_id_t,
                    ovrtx_timeout_t,
                    ctypes.POINTER(ovrtx_op_wait_result_t),
                ]
                lib.ovrtx_wait_op.restype = ovrtx_result_t

                lib.ovrtx_reset.argtypes = [ctypes.POINTER(ovrtx_renderer_t), ctypes.c_double]
                lib.ovrtx_reset.restype = ovrtx_enqueue_result_t

                lib.ovrtx_add_usd.argtypes = [
                    ctypes.POINTER(ovrtx_renderer_t),
                    ovrtx_usd_input_t,
                    ovx_string_t,
                    ctypes.POINTER(ovrtx_usd_handle_t),
                ]
                lib.ovrtx_add_usd.restype = ovrtx_enqueue_result_t

                lib.ovrtx_reset_stage.argtypes = [ctypes.POINTER(ovrtx_renderer_t)]
                lib.ovrtx_reset_stage.restype = ovrtx_enqueue_result_t

                lib.ovrtx_remove_usd.argtypes = [ctypes.POINTER(ovrtx_renderer_t), ovrtx_usd_handle_t]
                lib.ovrtx_remove_usd.restype = ovrtx_enqueue_result_t

                lib.ovrtx_update_stage_from_usd_time.argtypes = [ctypes.POINTER(ovrtx_renderer_t), ctypes.c_double]
                lib.ovrtx_update_stage_from_usd_time.restype = ovrtx_enqueue_result_t

                lib.ovrtx_clone_usd.argtypes = [
                    ctypes.POINTER(ovrtx_renderer_t),
                    ovx_string_t,
                    ctypes.POINTER(ovx_string_t),
                    ctypes.c_size_t,
                ]
                lib.ovrtx_clone_usd.restype = ovrtx_enqueue_result_t

                lib.ovrtx_step.argtypes = [
                    ctypes.POINTER(ovrtx_renderer_t),
                    ovrtx_render_product_set_t,
                    ctypes.c_double,
                    ctypes.POINTER(ovrtx_step_result_handle_t),
                ]
                lib.ovrtx_step.restype = ovrtx_enqueue_result_t

                lib.ovrtx_fetch_results.argtypes = [
                    ctypes.POINTER(ovrtx_renderer_t),
                    ovrtx_step_result_handle_t,
                    ovrtx_timeout_t,
                    ctypes.POINTER(ovrtx_render_product_set_outputs_t),
                ]
                lib.ovrtx_fetch_results.restype = ovrtx_result_t

                lib.ovrtx_destroy_results.argtypes = [
                    ctypes.POINTER(ovrtx_renderer_t),
                    ovrtx_step_result_handle_t,
                ]
                lib.ovrtx_destroy_results.restype = ovrtx_result_t

                lib.ovrtx_write_attribute.argtypes = [
                    ctypes.POINTER(ovrtx_renderer_t),
                    ctypes.POINTER(ovrtx_binding_desc_or_handle_t),
                    ctypes.POINTER(ovrtx_input_buffer_t),
                    ovrtx_data_access_t,
                ]
                lib.ovrtx_write_attribute.restype = ovrtx_enqueue_result_t

                lib.ovrtx_create_attribute_binding.argtypes = [
                    ctypes.POINTER(ovrtx_renderer_t),
                    ctypes.POINTER(ovrtx_binding_desc_t),
                    ctypes.POINTER(ovrtx_attribute_binding_handle_t),
                ]
                lib.ovrtx_create_attribute_binding.restype = ovrtx_enqueue_result_t

                lib.ovrtx_destroy_attribute_binding.argtypes = [
                    ctypes.POINTER(ovrtx_renderer_t),
                    ovrtx_attribute_binding_handle_t,
                ]
                lib.ovrtx_destroy_attribute_binding.restype = ovrtx_enqueue_result_t

                lib.ovrtx_map_attribute.argtypes = [
                    ctypes.POINTER(ovrtx_renderer_t),
                    ctypes.POINTER(ovrtx_binding_desc_or_handle_t),
                    ovrtx_mapping_desc_t,
                    ctypes.POINTER(ovrtx_attribute_mapping_t),
                ]
                lib.ovrtx_map_attribute.restype = ovrtx_result_t

                lib.ovrtx_unmap_attribute.argtypes = [
                    ctypes.POINTER(ovrtx_renderer_t),
                    ovrtx_map_handle_t,
                    ovrtx_cuda_sync_t,
                ]
                lib.ovrtx_unmap_attribute.restype = ovrtx_enqueue_result_t

                lib.ovrtx_map_rendered_output.argtypes = [
                    ctypes.POINTER(ovrtx_renderer_t),
                    ctypes.c_uint64,  # ovrtx_rendered_output_handle_t
                    ctypes.c_void_p,  # ovrtx_map_output_description_t* (specifies CPU or CUDA mapping)
                    ovrtx_timeout_t,  # timeout
                    ctypes.POINTER(ovrtx_rendered_output_t),  # out_rendered_output
                ]
                lib.ovrtx_map_rendered_output.restype = ovrtx_result_t

                lib.ovrtx_unmap_rendered_output.argtypes = [
                    ctypes.POINTER(ovrtx_renderer_t),
                    ctypes.c_uint64,  # ovrtx_rendered_output_map_handle_t
                    ovrtx_cuda_sync_t,  # before_destroy_cuda_sync
                ]
                lib.ovrtx_unmap_rendered_output.restype = ovrtx_result_t

                result = lib.ovrtx_initialize(ctypes.byref(config))
                if result.status != OVRTX_API_SUCCESS:
                    error = lib.ovrtx_get_last_error()
                    error_msg = str(error) if error.ptr else "Unknown error"
                    raise RuntimeError(f"Failed to initialize ovrtx library: {error_msg}")

                # Make sure to properly shutdown the library when the interpreter exits (__del__ is unreliable)
                import atexit

                self._lib = lib
                atexit.register(self._cleanup)

            except AttributeError as exc:
                raise RuntimeError(
                    f"Function not found in {lib._name if lib else self._lib_name}. Binary version may not match Python bindings."
                ) from exc

        return Bindings(self._lib)

    def _check_pxr_not_available(self) -> None:
        """Verify that the pxr package (usd-core) is not available.

        The ovrtx C library links to its own version of the USD libraries. Having
        the pxr Python package available can cause the C library to load an
        incompatible version of libusd, potentially leading to undefined behavior.

        Raises:
            RuntimeError: If pxr package is found or already imported.
        """
        if os.environ.get("OVRTX_SKIP_USD_CHECK", "0") == "1":
            return

        # Check if pxr was already imported by something else in this process
        if "pxr" in sys.modules:
            raise RuntimeError(
                "The pxr Python package (usd-core) is already imported. "
                "The ovrtx library cannot be loaded because it bundles its own USD libraries "
                "which would conflict with the already-loaded pxr version. "
                "Please use ovrtx in a Python environment without usd-core installed."
            )

        # Check if pxr is available (without importing it)
        import importlib.util

        if importlib.util.find_spec("pxr") is not None:
            raise RuntimeError(
                "The pxr Python package (usd-core) is installed in this environment. "
                "The ovrtx library cannot be loaded because it bundles its own USD libraries "
                "which would conflict with the pxr installation. "
                "Please use ovrtx in a Python environment without usd-core installed, "
                "or uninstall usd-core (pip uninstall usd-core)."
            )

    def _load_library(self, extra_paths: Optional[List[Path]] = None) -> ctypes.CDLL:
        """Load the library from search paths or raise an exception."""
        # Build search path list (extra paths first, then default paths)
        search_paths = []

        if extra_paths:
            search_paths.extend([p.resolve() for p in extra_paths if p.exists() and p.is_dir()])

        search_paths.extend(self._lib_search_paths)

        # Try loading from each path
        lib_paths = [p / self._lib_name for p in search_paths]
        last_error = None
        for lib_path in lib_paths:
            if lib_path.exists() and lib_path.is_file():
                try:
                    # Windows only: set OMNI_PLUGINS_BASE_PATH and OMNI_USD_PLUGINS_BASE_PATH if not
                    # already set. They're used to configure PATH, DLL directories and PXR_PLUGINPATH_NAME
                    # for USD plugin loading. The library's parent directory is the natural base path.
                    # Linux uses LD_LIBRARY_PATH/RPATH and doesn't need this workaround.
                    if sys.platform.startswith("win"):
                        if "OMNI_PLUGINS_BASE_PATH" not in os.environ:
                            os.environ["OMNI_PLUGINS_BASE_PATH"] = str(lib_path.parent)
                        if "OMNI_USD_PLUGINS_BASE_PATH" not in os.environ:
                            os.environ["OMNI_USD_PLUGINS_BASE_PATH"] = str(lib_path.parent)

                    lib = ctypes.CDLL(str(lib_path))
                    return lib
                except Exception as e:
                    last_error = e
                    continue

        # Throw with verbose message if we failed to load from any path
        paths_str = "\n  ".join(str(p) for p in lib_paths)
        error_msg = f"Failed to load {self._lib_name}. Tried:\n  {paths_str}"
        if last_error:
            error_msg += f"\nLast error: {last_error}"
        raise RuntimeError(error_msg)

    @property
    def is_loaded(self) -> bool:
        """Check if library is already loaded."""
        return self._lib is not None


_ovrtx_loader = _LibraryLoader()

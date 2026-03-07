# Copyright (c) 2025-2026, NVIDIA CORPORATION. All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

"""Math types for ovrtx (internal implementation).

This module provides math types like Matrix4d for use with ovrtx attribute operations.
"""

import ctypes

from .dlpack import DLDataType, DLDataTypeCode, DLDevice, DLDeviceType, DLTensor

__all__ = ["Matrix4d"]


class Matrix4d(ctypes.Structure):
    """4x4 double-precision matrix (compatible with usdrt::GfMatrix4d).

    Memory layout: 16 doubles in row-major order (128 bytes).
    Compatible with C++ usdrt::GfMatrix4d = omni::math::linalg::matrix4<double>.

    Example:
        ```python
        from ovrtx.math import Matrix4d

        # Create identity matrix
        m = Matrix4d()
        m.SetIdentity()

        # Access elements
        m[0][0] = 99.0

        # Use with write_attribute
        from ovrtx import Semantic
        renderer.write_attribute(
            prim_paths=["/World/Cube"],
            attribute_name="omni:xform",
            tensor=m.to_dltensor(),
            semantic=Semantic.XFORM_MAT4x4,
        )
        ```
    """

    _fields_ = [("v", (ctypes.c_double * 4) * 4)]
    _pack_ = 1  # Ensure no padding

    def __init__(self):
        """Initialize matrix to identity."""
        super().__init__()
        self.SetIdentity()

    def SetIdentity(self) -> None:
        """Set matrix to identity matrix."""
        for i in range(4):
            for j in range(4):
                self.v[i][j] = 1.0 if i == j else 0.0

    def SetTranslate(self, x: float, y: float, z: float) -> None:
        """Set matrix to translation matrix.

        Args:
            x: Translation in X direction
            y: Translation in Y direction
            z: Translation in Z direction
        """
        self.SetIdentity()
        self.v[3][0] = x
        self.v[3][1] = y
        self.v[3][2] = z

    def SetScale(self, s: float) -> None:
        """Set matrix to uniform scale matrix.

        Args:
            s: Scale factor
        """
        self.SetIdentity()
        for i in range(3):
            self.v[i][i] = s

    def __getitem__(self, row: int):
        """Access matrix row.

        Args:
            row: Row index (0-3)

        Returns:
            Row array (supports [row][col] indexing)
        """
        return self.v[row]

    def __setitem__(self, row: int, value):
        """Set matrix row.

        Args:
            row: Row index (0-3)
            value: Row data (list or array of 4 doubles)
        """
        if isinstance(value, (list, tuple)) and len(value) == 4:
            for col in range(4):
                self.v[row][col] = float(value[col])
        else:
            raise ValueError(f"Row value must be list/tuple of 4 floats, got {type(value)}")

    def __repr__(self) -> str:
        """Pretty print matrix."""
        rows = []
        for i in range(4):
            row_str = ", ".join(f"{self.v[i][j]:.6g}" for j in range(4))
            rows.append(f"({row_str})")
        return f"Matrix4d({', '.join(rows)})"

    def to_dltensor(self) -> DLTensor:
        """Create DLTensor for this matrix (for use with Renderer.write_attribute).

        Returns a DLTensor with the format expected by the C library:
        - shape[0] = 1 (one matrix)
        - dtype.lanes = 16 (16 doubles per matrix)
        - dtype.bits = 64
        - dtype.code = kDLFloat

        Returns:
            DLTensor pointing to this matrix's data
        """
        from .dlpack import DLTensor as DLTensorType

        dl_tensor = DLTensorType()
        dl_tensor.data = ctypes.cast(ctypes.pointer(self), ctypes.c_void_p)
        dl_tensor.device = DLDevice(device_type=DLDeviceType.kDLCPU, device_id=0)
        dl_tensor.ndim = 1
        dl_tensor.dtype = DLDataType(code=DLDataTypeCode.kDLFloat, bits=64, lanes=16)  # Matrix4d has 16 doubles
        dl_tensor.strides = None
        dl_tensor.byte_offset = 0

        # Create shape array: [1] (one matrix)
        # Keep reference on self to prevent GC
        shape_array = (ctypes.c_int64 * 1)()
        shape_array[0] = 1
        dl_tensor.shape = ctypes.cast(shape_array, ctypes.POINTER(ctypes.c_int64))

        # Keep shape array alive
        if not hasattr(self, "_dltensor_shape_refs"):
            self._dltensor_shape_refs = []
        self._dltensor_shape_refs.append(shape_array)

        return dl_tensor

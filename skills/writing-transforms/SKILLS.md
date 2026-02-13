<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: LicenseRef-NvidiaProprietary

NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
property and proprietary rights in and to this material, related
documentation and any modifications thereto. Any use, reproduction,
disclosure or distribution of this material and related documentation
without an express license agreement from NVIDIA CORPORATION or
its affiliates is strictly prohibited.
-->
---
name: writing-transforms
description: Writing 4x4 transform matrices to move, rotate, or scale prims. Use when user asks to move an object, set a transform, update position, animate a transform, or set a camera transform.
---

# Writing Transforms

## Overview

Transforms control the position, rotation, and scale of prims in the scene. ovrtx supports three transform representations. The most common is a 4x4 matrix of doubles using the USD row-vector convention: translation is in the last row (`[3][0..2]`). Transforms are written to the `omni:fabric:localMatrix` attribute.

## Python

### Write a 4x4 transform (single prim)

```python
from ovrtx.math import Matrix4d

matrix = Matrix4d()  # identity by default
matrix.SetTranslate(10.0, 0.0, 0.0)  # optional: set translation

binding = renderer.bind_attribute(
    prim_paths=["/World/Cube"],
    attribute_name="omni:fabric:localMatrix",
    semantic="transform_4x4",
)
binding.write(matrix.to_dltensor())
binding.unbind()
```

### Write transforms for multiple prims

```python
import numpy as np

binding = renderer.bind_attribute(
    prim_paths=[f"/World/Obj_{i}" for i in range(num_prims)],
    attribute_name="omni:fabric:localMatrix",
    semantic="transform_4x4",
)

with binding.map(device="cpu") as mapping:
    array = np.from_dlpack(mapping.tensor)  # shape: (num_prims, 4, 4)
    for i in range(num_prims):
        array[i] = np.eye(4)
        array[i][3, 0] = i * 10.0  # translate along X

binding.unbind()
```

## C

### Write identity transform on a single prim

```c
ovx_string_t path = {"/World/Cube", strlen("/World/Cube")};

// 4x4 identity (row-major, same as USD GfMatrix4d)
double transform[16] = {};
transform[0]  = 1.0;
transform[5]  = 1.0;
transform[10] = 1.0;
transform[15] = 1.0;

size_t count = 1;
DLDataType type = { kDLFloat, 64, 16 };  // 16 doubles per element

DLTensor tensor = {};
tensor.data = (void*)transform;
tensor.device = { kDLCPU, 0 };
tensor.ndim = 1;
tensor.dtype = type;
tensor.shape = (int64_t*)&count;

ovrtx_input_buffer_t buffer = { &tensor, 1, NULL, {} };

ovrtx_binding_desc_or_handle_t binding = {};
binding.binding_desc.prim_list = { &path, 1 };
binding.binding_desc.attribute_name = { {}, {
    "omni:fabric:localMatrix", strlen("omni:fabric:localMatrix") } };
binding.binding_desc.attribute_type = { type, false, OVRTX_SEMANTIC_TRANSFORM_4x4 };

ovrtx_write_attribute(renderer, &binding, &buffer, OVRTX_DATA_ACCESS_SYNC);
```

### Batch write (N prims)

```c
ovx_string_t paths[] = {
    {"/World/A", strlen("/World/A")},
    {"/World/B", strlen("/World/B")},
};
double transforms[2][16] = {};
// ... fill transforms ...

size_t count = 2;
DLDataType type = { kDLFloat, 64, 16 };

DLTensor tensor = {};
tensor.data = (void*)transforms;
tensor.device = { kDLCPU, 0 };
tensor.ndim = 1;
tensor.dtype = type;
tensor.shape = (int64_t*)&count;

ovrtx_input_buffer_t buffer = { &tensor, 1, NULL, {} };

ovrtx_binding_desc_or_handle_t binding = {};
binding.binding_desc.prim_list = { paths, count };
binding.binding_desc.attribute_name = { {}, {
    "omni:fabric:localMatrix", strlen("omni:fabric:localMatrix") } };
binding.binding_desc.attribute_type = { type, false, OVRTX_SEMANTIC_TRANSFORM_4x4 };

ovrtx_write_attribute(renderer, &binding, &buffer, OVRTX_DATA_ACCESS_SYNC);
```

## Key Types / Functions

| Representation | Semantic | DLDataType | Size per prim |
|----------------|----------|------------|---------------|
| 4x4 matrix (doubles) | `OVRTX_SEMANTIC_TRANSFORM_4x4` | `{ kDLFloat, 64, 16 }` | 128 bytes |
| pos3d + rot4f + scale3f | `OVRTX_SEMANTIC_TRANSFORM_POS3d_ROT4f_SCALE3f` | varies | 56 bytes |
| pos3d + rot3x3f | `OVRTX_SEMANTIC_TRANSFORM_POS3d_ROT3x3f` | varies | 64 bytes |

Python semantic strings: `"transform_4x4"`, `"transform"`, `"matrix"`

## Common Pitfalls

- Matrices use the **USD row-vector convention** (same as `GfMatrix4d`). Translation is in the last row: `v[12]`, `v[13]`, `v[14]` (or `matrix[3][0..2]` in Python/C).
- The attribute name is `"omni:fabric:localMatrix"` -- this is the Fabric-native transform attribute.
- Transform dtype is `float64` (doubles), not float32. Using the wrong dtype will cause errors.
- For repeated per-frame transform updates, use attribute bindings or mapping for better performance (see `attribute-bindings` and `mapping-attributes` skills).

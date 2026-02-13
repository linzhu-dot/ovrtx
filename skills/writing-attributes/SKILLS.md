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
name: writing-attributes
description: Writing scalar and array attribute data to prims. Use when user asks to write an attribute, set a property, change a material, set a color, or modify mesh data.
---

# Writing Attributes

## Overview

Beyond transforms, you can write arbitrary attributes to prims: colors, visibility, mesh geometry, string tokens, and more. ovrtx uses DLTensor as the data format, and the dtype must exactly match the USD attribute schema.

There are two categories:
- **Scalar attributes** -- one value per prim (e.g., a color, a transform)
- **Array attributes** -- variable-length per prim (e.g., mesh points, face vertex counts)

## Python

### Scalar attribute write (color)

```python
import numpy as np
from ovrtx._src.dlpack import DLTensor

# RGBA color (4 uint8 per prim)
colors = np.array([[255, 0, 0, 255]], dtype=np.uint8)  # red
tensor = DLTensor.from_dlpack(colors)

renderer.write_attribute(
    prim_paths=["/World/Cube"],
    attribute_name="primvars:displayColor",
    tensor=tensor,
    semantic="color_rgba4b",
)
```

### RGB float color

```python
colors = np.array([[1.0, 0.0, 0.0]], dtype=np.float32)  # red, 3 floats
tensor = DLTensor.from_dlpack(colors)

renderer.write_attribute(
    prim_paths=["/World/Cube"],
    attribute_name="primvars:displayColor",
    tensor=tensor,
    semantic="color_rgb3f",
)
```

### Array attribute write (mesh points)

```python
# float3[] points -- use float32, shape (N, 3)
points = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]], dtype=np.float32)

renderer.write_array_attribute(
    prim_paths=["/World/Mesh"],
    attribute_name="points",
    tensors=[DLTensor.from_dlpack(points)],
)
```

### Array attribute write (face vertex counts)

```python
# int[] faceVertexCounts -- use int32
face_counts = np.array([4, 4, 4], dtype=np.int32)

renderer.write_array_attribute(
    prim_paths=["/World/Mesh"],
    attribute_name="faceVertexCounts",
    tensors=[DLTensor.from_dlpack(face_counts)],
)
```

### Selective updates with dirty bits

```python
import numpy as np
from ovrtx._src.dlpack import DLTensor

# Only update prim 0 and prim 2 out of 3
values = np.array([1.0, 2.0, 3.0], dtype=np.float64)
tensor = DLTensor.from_dlpack(values)

dirty_bits = bytes([0b00000101])  # bits 0 and 2 set
renderer.write_attribute(
    prim_paths=["/World/A", "/World/B", "/World/C"],
    attribute_name="myCustomAttr",
    tensor=tensor,
    dirty_bits=dirty_bits,
)
```

### Prim modes

```python
import numpy as np
from ovrtx._src.dlpack import DLTensor

# "existing_only" (default) -- skip prims that don't exist
# "must_exist" -- error if any prim doesn't exist
# "create_new" -- create prim+attribute if it doesn't exist
values = DLTensor.from_dlpack(np.array([1.0], dtype=np.float32))
renderer.write_attribute(
    prim_paths=["/World/NewPrim"],
    attribute_name="myAttr",
    tensor=values,
    prim_mode="create_new",
)
```

## C

### Generic scalar write in C

```c
// Example: write a float3 color to a single prim
ovx_string_t prim_paths[] = { {"/World/Mesh", strlen("/World/Mesh")} };
float color[3] = { 1.0f, 0.0f, 0.0f };
size_t count = 1;
DLDataType type = { kDLFloat, 32, 3 };

DLTensor tensor = {};
tensor.data = (void*)color;
tensor.device = { kDLCPU, 0 };
tensor.ndim = 1;
tensor.dtype = type;
tensor.shape = (int64_t*)&count;

ovrtx_input_buffer_t buffer = { &tensor, 1, NULL, {} };

ovrtx_binding_desc_or_handle_t binding = {};
binding.binding_desc.prim_list = { prim_paths, count };
binding.binding_desc.attribute_name = { {}, {
    "primvars:displayColor", strlen("primvars:displayColor") } };
binding.binding_desc.attribute_type = { type, false, OVRTX_SEMANTIC_COLOR_RGB3f };

ovrtx_write_attribute(renderer, &binding, &buffer, OVRTX_DATA_ACCESS_SYNC);
```

## Key Types / Functions

| Python | C |
|--------|---|
| `renderer.write_attribute(...)` | `ovrtx_write_attribute(renderer, &binding, &buffer, access)` |
| `renderer.write_array_attribute(...)` | same, with `is_array=true` in binding desc |
| `semantic="color_rgba4b"` | `OVRTX_SEMANTIC_COLOR_RGBA4b` |
| `semantic="color_rgb3f"` | `OVRTX_SEMANTIC_COLOR_RGB3f` |

Semantics:
- `OVRTX_SEMANTIC_NONE` -- generic, dtype must be provided
- `OVRTX_SEMANTIC_TRANSFORM_4x4` -- 4x4 double matrix
- `OVRTX_SEMANTIC_COLOR_RGBA4b` -- 4 uint8
- `OVRTX_SEMANTIC_COLOR_RGB3f` -- 3 float32
- `OVRTX_SEMANTIC_PATH_STRING` -- prim path strings
- `OVRTX_SEMANTIC_TOKEN_STRING` -- token strings

Data access modes:
- `OVRTX_DATA_ACCESS_SYNC` -- copies data during the call, safe to free after return
- `OVRTX_DATA_ACCESS_ASYNC` -- data accessed later during stream execution, must keep alive

## Common Pitfalls

- Array attribute dtype must exactly match the USD schema. Using numpy's default `float64` for a `float3[]` attribute (which expects `float32`) will cause errors.
- String attributes (`path_string`, `token_string`) only support synchronous data access.
- For array attributes in Python, pass a list of DLTensors (one per prim), not a single tensor.
- `dirty_bits` is a bitvector with 1 bit per prim -- the byte array size must be `(prim_count + 7) / 8`.

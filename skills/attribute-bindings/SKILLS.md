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
name: attribute-bindings
description: Creating persistent attribute bindings for efficient repeated writes. Use when user asks about persistent bindings, repeated writes, efficient animation loops, or bind_attribute.
---

# Attribute Bindings

## Overview

When writing the same attribute to the same set of prims every frame (e.g., animating transforms), creating a persistent binding avoids the overhead of rebuilding the binding descriptor each time. Create the binding once, then call `write()` or `map()` on it repeatedly.

## Python

### Create binding and write repeatedly

```python
from ovrtx.math import Matrix4d

# Create binding once
binding = renderer.bind_attribute(
    prim_paths=["/World/Cube"],
    attribute_name="omni:fabric:localMatrix",
    semantic="transform_4x4",
    prim_mode="must_exist",
)

# Write every frame
for frame_idx in range(100):
    matrix = Matrix4d()
    matrix.SetTranslate(frame_idx * 0.1, 0.0, 0.0)
    binding.write(matrix.to_dltensor())
    renderer.step(render_products={"/Render/Camera"}, delta_time=1.0/60)

# Cleanup
binding.unbind()
```

### Array attribute binding

```python
from ovrtx._src.dlpack import DLDataType

binding = renderer.bind_array_attribute(
    prim_paths=["/World/Mesh1", "/World/Mesh2"],
    attribute_name="faceVertexCounts",
    dtype=DLDataType.from_str("int32"),
)

binding.write([counts_mesh1, counts_mesh2])  # list of tensors, one per prim
binding.unbind()
```

### Use binding for mapping (zero-copy)

```python
import numpy as np

binding = renderer.bind_attribute(
    prim_paths=all_prim_paths,
    attribute_name="omni:fabric:localMatrix",
    semantic="transform_4x4",
    prim_mode="must_exist",
)

for frame_idx in range(100):
    with binding.map(device="cpu") as mapping:
        array = np.from_dlpack(mapping.tensor)
        array[:] = compute_transforms(frame_idx)
    # unmap happens on context exit, flushing data to Fabric

    renderer.step(render_products={"/Render/Camera"}, delta_time=1.0/60)

binding.unbind()
```

## C

### Create and use a persistent binding

```c
// Create binding
ovrtx_binding_desc_t desc = {};
desc.prim_list.prim_paths = paths;
desc.prim_list.num_paths = path_count;
desc.attribute_name = (ovx_string_or_token_t){ {}, {
    "omni:fabric:localMatrix", strlen("omni:fabric:localMatrix") } };
desc.attribute_type = (ovrtx_attribute_type_t){ { kDLFloat, 64, 16 }, false, OVRTX_SEMANTIC_TRANSFORM_4x4 };
desc.prim_mode = OVRTX_BINDING_PRIM_MODE_MUST_EXIST;
desc.flags = OVRTX_BINDING_FLAG_OPTIMIZE;  // optimize for frequent writes

ovrtx_attribute_binding_handle_t binding_handle = 0;
ovrtx_enqueue_result_t result =
    ovrtx_create_attribute_binding(renderer, &desc, &binding_handle);

// Write using the handle
ovrtx_binding_desc_or_handle_t binding_ref = {};
binding_ref.binding_handle = binding_handle;

ovrtx_write_attribute(renderer, &binding_ref, &buffer, OVRTX_DATA_ACCESS_SYNC);

// Destroy when done
ovrtx_destroy_attribute_binding(renderer, binding_handle);
```

## Key Types / Functions

| Python | C |
|--------|---|
| `renderer.bind_attribute(...)` | `ovrtx_create_attribute_binding(renderer, &desc, &handle)` |
| `renderer.bind_array_attribute(...)` | same, with `is_array=true` in desc |
| `binding.write(tensor)` | `ovrtx_write_attribute(renderer, &binding_ref, &buffer, access)` |
| `binding.map(device=...)` | `ovrtx_map_attribute(renderer, &binding_ref, mapping_desc, &out)` |
| `binding.unbind()` | `ovrtx_destroy_attribute_binding(renderer, handle)` |

Binding flags:
- `OVRTX_BINDING_FLAG_NONE` -- default
- `OVRTX_BINDING_FLAG_OPTIMIZE` -- optimize internal structures for frequent high-volume writes

## Common Pitfalls

- Always call `unbind()` (Python) or `ovrtx_destroy_attribute_binding()` (C) when done. In Python, the binding auto-unbinds on garbage collection, but explicit cleanup is preferred.
- The binding locks in the prim list, attribute name, dtype, and semantic. You cannot change these after creation.
- `OVRTX_BINDING_FLAG_OPTIMIZE` should be used for the primary hot-path binding. The last binding created with this flag takes priority.
- In Python, `bind_attribute` is synchronous (blocks until the binding is ready). Use `bind_attribute_async` for non-blocking creation.

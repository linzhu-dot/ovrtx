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
name: mapping-attributes
description: Zero-copy attribute map/unmap for direct memory access to RTX internal buffers. Use when user asks about zero-copy writes, map attribute, direct memory access, Warp kernel writes, or GPU attribute updates.
---

# Mapping Attributes

## Overview

Mapping gives you direct access to RTX's internal Fabric buffer for an attribute. Instead of copying data in (like `write_attribute`), you write directly into the buffer, then unmap to apply changes. This is the most efficient path for per-frame updates, especially with GPU compute (e.g., Warp kernels).

The pattern is: **map -> write into tensor -> unmap**.

## Python

### CPU mapping with NumPy

```python
import numpy as np

mapping = renderer.map_attribute(
    prim_paths=["/World/Cube"],
    attribute_name="omni:fabric:localMatrix",
    semantic="transform_4x4",
    device="cpu",
)

# Write directly into the mapped buffer
array = np.from_dlpack(mapping.tensor)  # shape: (1, 4, 4) for 1 prim
array[0] = np.eye(4)

# Apply and release
renderer.unmap_attribute(mapping)
```

### Context manager (recommended)

```python
with renderer.map_attribute(
    prim_paths=["/World/Cube"],
    attribute_name="omni:fabric:localMatrix",
    semantic="transform_4x4",
) as mapping:
    np.from_dlpack(mapping.tensor)[0] = np.eye(4)
# Automatically unmapped on exit
```

### GPU mapping with Warp

From the planet-system example -- map on CUDA, compute with a Warp kernel, unmap with stream sync:

```python
import warp as wp

with binding.map(device="cuda", device_id=0) as attr_mapping:
    wp_transforms = wp.from_dlpack(attr_mapping.tensor, dtype=wp.mat44d)

    wp.launch(
        kernel=compute_transforms,
        dim=num_prims,
        inputs=[wp_transforms, wp.float64(sim_time)],
        device=wp_transforms.device,
    )

    # Sync: tell ovrtx to wait for this stream before reading the data
    attr_mapping.unmap(stream=cuda_stream.cuda_stream)
```

### Using a persistent binding for mapping

More efficient for repeated map/unmap cycles (avoids recreating the binding descriptor):

```python
binding = renderer.bind_attribute(
    prim_paths=prim_paths,
    attribute_name="omni:fabric:localMatrix",
    semantic="transform_4x4",
    prim_mode="must_exist",
)

for frame in range(num_frames):
    with binding.map(device="cpu") as mapping:
        array = np.from_dlpack(mapping.tensor)
        array[:] = frame_transforms[frame]

    renderer.step(render_products={"/Render/Camera"}, delta_time=dt)

binding.unbind()
```

## C

### Map, write, unmap

```c
ovrtx_mapping_desc_t mapping_desc = {};
mapping_desc.device_type = 1;  // kDLCPU
mapping_desc.device_id = 0;

// Build the binding descriptor (or use a persistent binding handle,
// see attribute-bindings skill)
ovx_string_t paths[] = { {"/World/Cube", strlen("/World/Cube")} };
DLDataType type = { kDLFloat, 64, 16 };  // 16 doubles per element

ovrtx_binding_desc_or_handle_t binding = {};
binding.binding_desc.prim_list = { paths, 1 };
binding.binding_desc.attribute_name = { {}, {
    "omni:fabric:localMatrix", strlen("omni:fabric:localMatrix") } };
binding.binding_desc.attribute_type = { type, false, OVRTX_SEMANTIC_TRANSFORM_4x4 };

ovrtx_attribute_mapping_t out_mapping = {};
ovrtx_result_t result = ovrtx_map_attribute(
    renderer, &binding, mapping_desc, &out_mapping);

// Write into out_mapping.dl (DLTensor)
double* data = (double*)out_mapping.dl.data;
// ... write transform data ...

// Unmap to apply
ovrtx_cuda_sync_t no_sync = {};
ovrtx_unmap_attribute(renderer, out_mapping.map_handle, no_sync);
```

### GPU mapping with CUDA sync

```c
ovrtx_mapping_desc_t mapping_desc = {};
mapping_desc.device_type = 2;  // kDLCUDA
mapping_desc.device_id = 0;

ovrtx_attribute_mapping_t out_mapping = {};
ovrtx_result_t result = ovrtx_map_attribute(
    renderer, &binding, mapping_desc, &out_mapping);

// Launch CUDA kernel to write into out_mapping.dl.data on cuda_stream
// ...

// Signal event when kernel is done
cuEventRecord(write_done_event, cuda_stream);

// Unmap with sync -- ovrtx waits for the event before reading the data
ovrtx_cuda_sync_t sync = {};
sync.wait_event = (uintptr_t)write_done_event;
ovrtx_unmap_attribute(renderer, out_mapping.map_handle, sync);
```

## Key Types / Functions

| Python | C |
|--------|---|
| `renderer.map_attribute(...)` | `ovrtx_map_attribute(renderer, &binding, desc, &out)` |
| `renderer.unmap_attribute(mapping)` | `ovrtx_unmap_attribute(renderer, handle, sync)` |
| `binding.map(device=...)` | same, with binding handle |
| `mapping.tensor` | `out_mapping.dl` (DLTensor) |
| `mapping.unmap(stream=...)` | `ovrtx_unmap_attribute` with `cuda_sync.stream` |
| `mapping.unmap(event=...)` | `ovrtx_unmap_attribute` with `cuda_sync.wait_event` |

Semantic-aware tensor shapes in Python:
- `TRANSFORM_4x4`: reshaped to (N, 4, 4) for NumPy-friendly matrix operations
- `COLOR_RGBA4b`: reshaped to (N, 4) for RGBA channels
- `COLOR_RGB3f`: reshaped to (N, 3) for RGB channels

## Common Pitfalls

- Array attributes (e.g., `float3[] points`) are **not supported** for mapping because they have variable lengths per prim. Use `write_array_attribute()` instead.
- Data must be fully written before calling `unmap()`. For CUDA, pass `stream` or `event` so ovrtx knows when the GPU write is complete.
- `event` and `stream` are mutually exclusive on unmap -- use one or the other.
- Providing `event`/`stream` for a CPU-mapped attribute raises an error.
- Multiple mappings can be outstanding on the same attribute. The logical order of application depends on the order of `unmap()` calls.

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
name: reading-render-output
description: Mapping rendered output to access pixel data on CPU or GPU. Use when user asks to get pixels, read an image, save a PNG, display rendered output, or access render var data.
---

# Reading Render Output

## Overview

After stepping the renderer and fetching results, each `RenderVarOutput` (e.g., `LdrColor`, `HdrColor`, `depth`) must be mapped to access its pixel data.

- In Python, render-var mapping supports `device="cpu"` and `device="cuda"`.
- In C, `ovrtx_map_rendered_output` also supports `OVRTX_MAP_DEVICE_TYPE_CUDA_ARRAY` for CUDA-array workflows.

After reading/processing, unmap to release the mapped buffer.

## Python

### Map to CPU and read as NumPy

```python
products = renderer.step(render_products={"/Render/Camera"}, delta_time=1.0/60)

for product_name, product in products.items():
    for frame in product.frames:
        with frame.render_vars["LdrColor"].map(device="cpu") as var:
            pixels = var.tensor.numpy()  # shape: (H, W, 4), dtype: uint8
```

### Save as PNG with Pillow

```python
from PIL import Image

with frame.render_vars["LdrColor"].map(device="cpu") as var:
    pixels = var.tensor.numpy()
    Image.fromarray(pixels).save("output.png")
```

### Map to CUDA for GPU processing

```python
import warp as wp

# Example stream setup (e.g. Warp stream):
# cuda_stream = wp.Stream(device="cuda:0")
with frame.render_vars["LdrColor"].map(device="cuda") as mapping:
    wp_image = wp.from_dlpack(mapping.tensor)
    # GPU operations on wp_image...
    mapping.unmap(stream=cuda_stream.cuda_stream)  # sync before release
```

## C

### Map to CPU

```c
ovrtx_map_output_description_t map_desc = {};
map_desc.device_type = OVRTX_MAP_DEVICE_TYPE_CPU;

ovrtx_rendered_output_t rendered_output = {};
ovrtx_result_t result = ovrtx_map_rendered_output(
    renderer, output_handle, &map_desc, ovrtx_timeout_infinite, &rendered_output);

// Access pixel data via DLTensor
DLTensor const& tensor = rendered_output.buffer.dl;
int width  = (int)tensor.shape[1];
int height = (int)tensor.shape[0];
void* pixels = tensor.data;

// Save with stb_image_write
stbi_write_png("out.png", width, height, 4, pixels, 4 * width);

// Unmap when done
ovrtx_cuda_sync_t no_sync = {};
ovrtx_unmap_rendered_output(renderer, rendered_output.map_handle, no_sync);
```

### Map to CUDA array (zero-copy)

```c
ovrtx_map_output_description_t map_desc = {};
map_desc.device_type = OVRTX_MAP_DEVICE_TYPE_CUDA_ARRAY;
map_desc.sync_stream = 0;  // no stream sync on map

ovrtx_rendered_output_t rendered_output = {};
ovrtx_result_t result = ovrtx_map_rendered_output(
    renderer, output_handle, &map_desc, ovrtx_timeout_infinite, &rendered_output);

// Wait for render completion before accessing
CUevent wait_event = (CUevent)rendered_output.buffer.cuda_sync.wait_event;
if (wait_event) {
    cuStreamWaitEvent(cuda_stream, wait_event, 0);
}

CUarray cuda_array = (CUarray)rendered_output.buffer.dl.data;
// Use cuda_array for CUDA operations...

// Signal when CUDA is done, then unmap
cuEventRecord(copy_done_event, cuda_stream);
ovrtx_cuda_sync_t sync = {};
sync.wait_event = (uintptr_t)copy_done_event;
ovrtx_unmap_rendered_output(renderer, rendered_output.map_handle, sync);
```

### Find a specific render var in C

This helper is not part of the ovrtx API; define it in your own code (see `examples/c/minimal/main.cpp`):

```c
ovrtx_rendered_output_handle_t find_output(
    ovrtx_render_product_set_outputs_t const& outputs,
    char const* name)
{
    for (size_t i = 0; i < outputs.output_count; ++i) {
        ovrtx_render_product_output_t const& product = outputs.outputs[i];
        for (size_t f = 0; f < product.output_frame_count; ++f) {
            ovrtx_render_product_frame_output_t const& frame = product.output_frames[f];
            for (size_t v = 0; v < frame.render_var_count; ++v) {
                ovrtx_render_product_render_var_output_t const& var = frame.output_render_vars[v];
                if (var.render_var_name.ptr &&
                    strncmp(var.render_var_name.ptr, name, var.render_var_name.length) == 0) {
                    return var.output_handle;
                }
            }
        }
    }
    return OVRTX_INVALID_HANDLE;
}
```

## Key Types / Functions

| Python | C |
|--------|---|
| `render_var.map(device="cpu")` | `ovrtx_map_rendered_output(renderer, handle, &desc, timeout, &output)` |
| `mapping.tensor.numpy()` | `rendered_output.buffer.dl` (DLTensor) |
| context manager `__exit__` | `ovrtx_unmap_rendered_output(renderer, map_handle, sync)` |
| `mapping.unmap(stream=...)` | `ovrtx_unmap_rendered_output` with `cuda_sync.stream` / `cuda_sync.wait_event` |

Device types:
- `OVRTX_MAP_DEVICE_TYPE_CPU` -- read back to host memory (sync + copy)
- `OVRTX_MAP_DEVICE_TYPE_CUDA` -- linear CUDA device memory (may copy)
- `OVRTX_MAP_DEVICE_TYPE_CUDA_ARRAY` -- CUDA array, zero-copy for image outputs
- `OVRTX_MAP_DEVICE_TYPE_DEFAULT` -- most efficient format

Common render vars:
- `LdrColor` -- RGBA uint8, sRGB color space
- `HdrColor` -- RGBA float16, linear color space

## Common Pitfalls

- Mapped output lifetime is independent of `ovrtx_destroy_results()`. Still unmap each mapped output promptly to avoid leaks.
- For `CUDA_ARRAY` mapping, you must wait on `rendered_output.buffer.cuda_sync.wait_event` before accessing the data.
- When unmapping after CUDA work, pass a `cuda_sync` so ovrtx knows when it's safe to reclaim the buffer.
- In Python, the context manager handles unmap automatically. For explicit CUDA sync, call `mapping.unmap(stream=...)` before exiting the `with` block.

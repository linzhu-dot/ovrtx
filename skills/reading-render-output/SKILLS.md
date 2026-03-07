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

- In Python, render-var mapping supports `device=Device.CPU` and `device=Device.CUDA` (from `from ovrtx import Device`).
- In C, `ovrtx_map_rendered_output` also supports `OVRTX_MAP_DEVICE_TYPE_CUDA_ARRAY` for zero-copy workflows.

After reading/processing, unmap to release the mapped buffer.

## Python

### Map to CPU and read as NumPy

> **Source:** `examples/python/minimal/main.py` snippet `read-render-output`

### Save as PNG with Pillow

> **Source:** `tests/test_ovrtx.py` snippet `step-and-read-output`

### Map to CUDA for GPU processing

> **Source:** `tests/test_ovrtx.py` snippet `unmap-attribute-cuda-sync`
>
> For render output CUDA mapping, see the `map-rendered-output-cuda` pattern in the same test file's `test_renderer_cuda`.

## C

### Map to CPU

> **Source:** `examples/c/minimal/main.cpp` snippet `map-rendered-output-cpu`

### Map to CUDA array (zero-copy)

> **Source:** `examples/c/vulkan-interop/src/main.cpp` snippet `map-rendered-output-cuda-array`

### Find a specific render var in C

This helper is not part of the ovrtx API; define it in your own code (see `examples/c/minimal/main.cpp`):

> **Source:** `examples/c/minimal/main.cpp` snippet `find-output-helper`

## Key Types / Functions

| Python | C |
|--------|---|
| `render_var.map(device=Device.CPU)` | `ovrtx_map_rendered_output(renderer, handle, &desc, timeout, &output)` |
| `mapping.tensor.numpy()` | `rendered_output.buffer.dl` (DLTensor) |
| context manager `__exit__` | `ovrtx_unmap_rendered_output(renderer, map_handle, sync)` |
| `mapping.unmap(stream=...)` | `ovrtx_unmap_rendered_output` with `cuda_sync.stream` / `cuda_sync.wait_event` |

Device types (Python: `from ovrtx import Device`):
- `Device.CPU` -- read back to host memory (sync + copy)
- `Device.CUDA` -- linear CUDA device memory (may copy)
- C also supports: `OVRTX_MAP_DEVICE_TYPE_CUDA_ARRAY` (zero-copy for image outputs) and `OVRTX_MAP_DEVICE_TYPE_DEFAULT` (auto-selects the most efficient format; returns a CUDA array for image outputs, avoiding an extra copy).

Common render vars:
- `LdrColor` -- RGBA uint8, sRGB color space
- `HdrColor` -- RGBA float16, linear color space

## Common Pitfalls

- Mapped output lifetime is independent of `ovrtx_destroy_results()`. Still unmap each mapped output promptly to avoid leaks.
- For `CUDA_ARRAY` mapping, you must wait on `rendered_output.buffer.cuda_sync.wait_event` before accessing the data.
- When unmapping after CUDA work, pass a `cuda_sync` so ovrtx knows when it's safe to reclaim the buffer.
- In Python, the context manager handles unmap automatically. For explicit CUDA sync, call `mapping.unmap(stream=...)` before exiting the `with` block.

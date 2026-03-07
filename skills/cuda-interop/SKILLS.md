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
name: cuda-interop
description: GPU interop patterns for CUDA arrays, timeline semaphores, and Vulkan shared memory. Use when user asks about CUDA interop, GPU rendering pipelines, Vulkan interop, shared memory, or timeline semaphores.
---

# CUDA Interop

## Overview

ovrtx renders on the GPU and can provide output as CUDA device memory, and in C it can also provide CUDA arrays (zero-copy). For advanced pipelines (e.g., Vulkan display, custom CUDA post-processing), you need to handle CUDA synchronization correctly to avoid race conditions between ovrtx's internal GPU work and your own.

## Python

### Map render output to CUDA

> **Source:** `tests/test_ovrtx.py` snippet `unmap-attribute-cuda-sync`
>
> For render output: use `frame.render_vars["LdrColor"].map(device=Device.CUDA)` instead of `renderer.map_attribute(...)`.

### Map attribute to CUDA for Warp kernel writes

> **Source:** `tests/test_ovrtx.py` snippet `map-attribute-gpu`

## C

`OVRTX_MAP_DEVICE_TYPE_CUDA_ARRAY` is available through the C API (`ovrtx_map_rendered_output`). Python render-var mapping uses `device=Device.CPU` / `device=Device.CUDA` and does not expose a CUDA-array selector.

### Map render output as CUDA array (zero-copy)

> **Source:** `examples/c/vulkan-interop/src/main.cpp` snippet `map-rendered-output-cuda-array`

### Wait for render completion before accessing

The output may not be fully written when `map` returns. Check the wait event:

> **Source:** `examples/c/vulkan-interop/src/main.cpp` snippet `map-rendered-output-cuda-array`
>
> Check `rendered_output.buffer.cuda_sync.wait_event` and call `cuStreamWaitEvent` before accessing.

### Signal when CUDA work is done, then unmap

> **Source:** `examples/c/vulkan-interop/src/main.cpp` snippet `write-camera-transform`
>
> Record a CUDA event after your kernel, then pass it via `ovrtx_cuda_sync_t.wait_event` on unmap.

### Double-buffered async pattern (from vulkan-interop example)

For maximum throughput, use two shared images and ping-pong between them.
CUDA writes to one while a consumer (e.g., Vulkan) reads the other.

> **Source:** `examples/c/vulkan-interop/src/main.cpp`
>
> See the full vulkan-interop example for the double-buffered async rendering pattern with timeline semaphores and CUDA-Vulkan shared images.

### Map with sync_stream (auto-sync)

If you provide `sync_stream` on the map call, ovrtx inserts the wait automatically:

> **Source:** `examples/c/vulkan-interop/src/main.cpp` snippet `map-rendered-output-cuda-array`
>
> Set `map_desc.sync_stream = (uintptr_t)cuda_stream` for automatic synchronization.

## Key Types / Functions

| Concept | Python | C |
|---------|--------|---|
| Map to CUDA | `render_var.map(device=Device.CUDA)` | `OVRTX_MAP_DEVICE_TYPE_CUDA` |
| Map to CUDA array | N/A (Python uses CUDA) | `OVRTX_MAP_DEVICE_TYPE_CUDA_ARRAY` |
| Wait for render | automatic via context manager | `cuStreamWaitEvent(stream, wait_event, 0)` |
| Signal done | `mapping.unmap(stream=...)` | `cuda_sync.wait_event = (uintptr_t)event` |
| Stream sync on map | `sync_stream` parameter | `map_desc.sync_stream` |

C CUDA sync struct:
```c
typedef struct {
    uintptr_t stream;     // 0 = none, 1 = default, >1 = specific stream
    uintptr_t wait_event; // CUevent to wait on (0 = none)
} ovrtx_cuda_sync_t;
```

## Common Pitfalls

- Always wait on `rendered_output.buffer.cuda_sync.wait_event` before accessing CUDA array data.
- Always signal completion (via event or stream) when unmapping after CUDA work, or ovrtx may reclaim the buffer while your kernel is still running.
- In C, `OVRTX_MAP_DEVICE_TYPE_CUDA_ARRAY` returns a `CUarray` (opaque pointer in `dl.data`), not linear memory. Use `surf2Dwrite`/`tex2D` to access it.
- In C, `OVRTX_MAP_DEVICE_TYPE_CUDA` returns linear device memory (may incur a copy for image outputs).
- In Python, `event` and `stream` on `unmap()` are mutually exclusive -- pass one or the other.
- `write_attribute()`, `write_array_attribute()`, and `binding.write()` also accept `cuda_stream=` and `cuda_event=` for GPU-synchronised writes with `data_access=DataAccess.ASYNC`. These are mutually exclusive, same as on `unmap()`.

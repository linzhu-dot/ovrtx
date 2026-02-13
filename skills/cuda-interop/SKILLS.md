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

```python
import warp as wp

products = renderer.step(render_products={"/Render/Camera"}, delta_time=1.0/60)

# Example stream setup:
# cuda_stream = wp.Stream(device="cuda:0")
for product_name, product in products.items():
    for frame in product.frames:
        with frame.render_vars["LdrColor"].map(device="cuda") as mapping:
            wp_image = wp.from_dlpack(mapping.tensor)
            # GPU operations on wp_image...

            # Explicit sync: tell ovrtx to wait for this stream
            mapping.unmap(stream=cuda_stream.cuda_stream)
```

### Map attribute to CUDA for Warp kernel writes

```python
with binding.map(device="cuda", device_id=0) as attr_mapping:
    wp_transforms = wp.from_dlpack(attr_mapping.tensor, dtype=wp.mat44d)

    stream = wp.Stream(device=wp_transforms.device)
    wp.launch(
        kernel=compute_transforms,
        dim=num_prims,
        inputs=[wp_transforms, wp.float64(time)],
        device=wp_transforms.device,
    )

    attr_mapping.unmap(stream=stream.cuda_stream)
```

## C

`OVRTX_MAP_DEVICE_TYPE_CUDA_ARRAY` is available through the C API (`ovrtx_map_rendered_output`). Python render-var mapping uses `device="cpu"` / `device="cuda"` and does not expose a CUDA-array selector.

### Map render output as CUDA array (zero-copy)

```c
ovrtx_map_output_description_t map_desc = {};
map_desc.device_type = OVRTX_MAP_DEVICE_TYPE_CUDA_ARRAY;
map_desc.sync_stream = 0;  // don't auto-sync on map

ovrtx_rendered_output_t rendered_output = {};
ovrtx_result_t result = ovrtx_map_rendered_output(
    renderer, output_handle, &map_desc, ovrtx_timeout_infinite, &rendered_output);
```

### Wait for render completion before accessing

The output may not be fully written when `map` returns. Check the wait event:

```c
CUevent wait_event = (CUevent)rendered_output.buffer.cuda_sync.wait_event;
if (wait_event) {
    cuStreamWaitEvent(cuda_stream, wait_event, 0);
}

// Now safe to access the CUDA array on cuda_stream
CUarray cuda_array = (CUarray)rendered_output.buffer.dl.data;
int width  = (int)rendered_output.buffer.dl.shape[1];
int height = (int)rendered_output.buffer.dl.shape[0];
```

### Signal when CUDA work is done, then unmap

```c
// After your CUDA kernel/copy finishes, record an event
cuEventRecord(copy_done_event, cuda_stream);

// Tell ovrtx to wait for this event before reclaiming the buffer
ovrtx_cuda_sync_t sync = {};
sync.wait_event = (uintptr_t)copy_done_event;
ovrtx_unmap_rendered_output(renderer, rendered_output.map_handle, sync);
```

### Double-buffered async pattern (from vulkan-interop example)

For maximum throughput, use two shared images and ping-pong between them.
CUDA writes to one while a consumer (e.g., Vulkan) reads the other.
See `examples/c/vulkan-interop/src/main.cpp` for the full implementation.

```c
// Setup: two shared surfaces, a timeline semaphore, and double-buffer state
CUsurfObject surfaces[2];      // imported from Vulkan exportable images
CUevent      frame_done_event; // poll CUDA completion without blocking
CUstream     cuda_stream;
int          write_idx = 0;
int          read_idx  = 0;
uint64_t     frame_counter = 0;
bool         cuda_work_pending = false;

// Render loop
while (running) {
    // 1. Non-blocking poll: did the previous CUDA frame finish?
    if (cuda_work_pending) {
        if (cuEventQuery(frame_done_event) == CUDA_SUCCESS) {
            std::swap(read_idx, write_idx);
            cuda_work_pending = false;
        }
    }

    // 2. Consumer reads surfaces[read_idx] (stale-by-one-frame, no wait needed)
    //    e.g., Vulkan samples the shared image as a texture

    // 3. If no CUDA work pending, kick off the next frame
    if (!cuda_work_pending) {
        // Step ovrtx and get the rendered output
        ovrtx_step_result_handle_t step_handle = 0;
        ovrtx_enqueue_result_t enqueue =
            ovrtx_step(renderer, render_products, delta_time, &step_handle);
        ovrtx_op_wait_result_t wait_result;
        ovrtx_wait_op(renderer, enqueue.op_index,
                      ovrtx_timeout_infinite, &wait_result);

        ovrtx_render_product_set_outputs_t outputs = {};
        ovrtx_fetch_results(renderer, step_handle,
                            ovrtx_timeout_infinite, &outputs);

        // Map as CUDA array (zero-copy)
        ovrtx_map_output_description_t map_desc = {};
        map_desc.device_type = OVRTX_MAP_DEVICE_TYPE_CUDA_ARRAY;

        ovrtx_rendered_output_t rendered_output = {};
        ovrtx_map_rendered_output(renderer, output_handle,
                                  &map_desc, ovrtx_timeout_infinite,
                                  &rendered_output);

        // Wait for ovrtx render to finish on our stream
        CUevent wait_event =
            (CUevent)rendered_output.buffer.cuda_sync.wait_event;
        if (wait_event) {
            cuStreamWaitEvent(cuda_stream, wait_event, 0);
        }

        // Copy from ovrtx CUDA array into shared surface[write_idx]
        CUarray cuda_array = (CUarray)rendered_output.buffer.dl.data;
        // ... surf2Dwrite or cuMemcpy2D from cuda_array to surfaces[write_idx] ...

        // Release ovrtx output (no sync needed -- we haven't given ovrtx
        // a stream/event, and our copy is on cuda_stream)
        ovrtx_cuda_sync_t no_sync = {};
        ovrtx_unmap_rendered_output(renderer, rendered_output.map_handle,
                                    no_sync);
        ovrtx_destroy_results(renderer, step_handle);

        // Signal completion
        cuEventRecord(frame_done_event, cuda_stream);
        frame_counter++;
        // Optionally signal a timeline semaphore for cross-API sync:
        // cuda_signal_timeline(frame_counter, cuda_stream);
        cuda_work_pending = true;
    }
}
```

Key elements:
1. Two shared images (e.g., Vulkan exportable images imported into CUDA via `cuImportExternalMemory`)
2. `cuEventQuery` to poll CUDA frame completion without blocking the CPU
3. Buffer swap (`std::swap(read_idx, write_idx)`) only when CUDA confirms the previous frame is done
4. For cross-API sync (e.g., Vulkan), a timeline semaphore shared via `cuImportExternalSemaphore`

### Map with sync_stream (auto-sync)

If you provide `sync_stream` on the map call, ovrtx inserts the wait automatically:

```c
ovrtx_map_output_description_t map_desc = {};
map_desc.device_type = OVRTX_MAP_DEVICE_TYPE_CUDA;
map_desc.sync_stream = (uintptr_t)cuda_stream;  // auto-sync

// After this call, any work on cuda_stream can safely access the output
ovrtx_map_rendered_output(renderer, output_handle, &map_desc, timeout, &rendered_output);
```

## Key Types / Functions

| Concept | Python | C |
|---------|--------|---|
| Map to CUDA | `render_var.map(device="cuda")` | `OVRTX_MAP_DEVICE_TYPE_CUDA` |
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
- `OVRTX_MAP_DEVICE_TYPE_CUDA_ARRAY` returns a `CUarray` (opaque pointer in `dl.data`), not linear memory. Use `surf2Dwrite`/`tex2D` to access it.
- `OVRTX_MAP_DEVICE_TYPE_CUDA` returns linear device memory (may incur a copy for image outputs).
- In Python, `event` and `stream` on `unmap()` are mutually exclusive -- pass one or the other.

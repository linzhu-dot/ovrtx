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
name: stepping-and-rendering
description: Running a simulation step to produce rendered frames. Use when user asks to render a frame, step the renderer, simulate a sensor, or get an image from ovrtx.
---

# Stepping and Rendering

## Overview

After loading USD content, you render frames by calling `step()`. Each step advances the simulation by `delta_time` and produces output for the specified render products (cameras/sensors). The result contains a hierarchy of products, frames, and render variables that can be mapped to access pixel data.

## Python

### Single frame

```python
products = renderer.step(
    render_products={"/Render/Camera"},
    delta_time=1.0 / 60  # 60 Hz
)
```

### Iterate over results

```python
for product_name, product in products.items():
    for frame in product.frames:
        for var_name, render_var in frame.render_vars.items():
            print(f"{product_name} / {var_name}")
```

### Render loop

```python
for i in range(100):
    products = renderer.step(
        render_products={"/Render/Camera"},
        delta_time=1.0 / 60
    )
    # process products...
```

### Reset simulation time

```python
renderer.reset(time=0.0)  # Reset accumulated time to 0
```

## C

### Step, wait, fetch

```c
// 1. Define render products
ovx_string_t rp_path = {"/Render/Camera", strlen("/Render/Camera")};
ovrtx_render_product_set_t render_products = {};
render_products.render_products = &rp_path;
render_products.num_render_products = 1;

// 2. Enqueue step
ovrtx_step_result_handle_t step_result_handle = 0;
ovrtx_enqueue_result_t enqueue_result =
    ovrtx_step(renderer, render_products, 1.0 / 60.0, &step_result_handle);

// 3. Wait for completion
ovrtx_op_wait_result_t wait_result;
ovrtx_result_t result = ovrtx_wait_op(
    renderer, enqueue_result.op_index, ovrtx_timeout_infinite, &wait_result);

// 4. Fetch results
ovrtx_render_product_set_outputs_t outputs = {};
result = ovrtx_fetch_results(
    renderer, step_result_handle, ovrtx_timeout_infinite, &outputs);

// 5. Process outputs (see reading-render-output skill)
// ...

// 6. Destroy results when done
ovrtx_destroy_results(renderer, step_result_handle);
```

### Iterate over C results

```c
for (size_t i = 0; i < outputs.output_count; ++i) {
    ovrtx_render_product_output_t const& product = outputs.outputs[i];
    for (size_t f = 0; f < product.output_frame_count; ++f) {
        ovrtx_render_product_frame_output_t const& frame = product.output_frames[f];
        for (size_t v = 0; v < frame.render_var_count; ++v) {
            ovrtx_render_product_render_var_output_t const& var = frame.output_render_vars[v];
            // var.render_var_name, var.output_handle
        }
    }
}
```

## Key Types / Functions

| Python | C |
|--------|---|
| `renderer.step(products, dt)` | `ovrtx_step()` + `ovrtx_wait_op()` + `ovrtx_fetch_results()` |
| `renderer.step_async(products, dt)` | `ovrtx_step()` (always async) |
| `renderer.reset(time)` | `ovrtx_reset(renderer, time)` |
| `RenderProductSetOutputs` | `ovrtx_render_product_set_outputs_t` |
| `ProductOutput` | `ovrtx_render_product_output_t` |
| `FrameOutput` | `ovrtx_render_product_frame_output_t` |
| `RenderVarOutput` | `ovrtx_render_product_render_var_output_t` |

Result hierarchy:
```
RenderProductSetOutputs
  -> ProductOutput (one per render product path)
    -> FrameOutput (one per frame produced during the step)
      -> RenderVarOutput (one per output variable: LdrColor, HdrColor, depth, etc.)
```

## Common Pitfalls

- In C, you must call `ovrtx_destroy_results()` after processing to free resources. ovrtx will warn if results are leaked.
- `delta_time` controls sensor simulation timing. A camera without motion blur produces one frame per step regardless of delta.
- The render product paths must match actual RenderProduct prims in the USD stage.
- In Python, `RenderProductSetOutputs` auto-destroys on garbage collection, but you can use it as a context manager for explicit control: `with renderer.step(...) as products:`.

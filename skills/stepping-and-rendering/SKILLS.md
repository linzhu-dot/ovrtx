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

> **Source:** `examples/python/minimal/main.py` snippet `step`

### Iterate over results

> **Source:** `examples/python/minimal/main.py` snippet `read-render-output`

### Render loop

> **Source:** `tests/test_ovrtx.py` snippet `step-and-read-output`

### Reset simulation time

> **Source:** `tests/test_ovrtx.py` snippet `reset-stage`
>
> `renderer.reset(time=0.0)` resets accumulated simulation time (distinct from `reset_stage()`).

## C

### Step, wait, fetch

> **Source:** `examples/c/minimal/main.cpp` snippet `step-renderer`
>
> Followed by: `examples/c/minimal/main.cpp` snippet `fetch-results`

### Iterate over C results

> **Source:** `examples/c/minimal/main.cpp` snippet `find-output-helper`

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

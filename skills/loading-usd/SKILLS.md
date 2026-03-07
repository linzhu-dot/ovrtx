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
name: loading-usd
description: Loading USD scenes into the renderer from files, URLs, or inline USDA strings. Use when user asks to load a USD scene, add a layer, compose USD content, or create runtime geometry.
---

# Loading USD

## Overview

Before rendering, you must load USD content into the renderer. ovrtx supports three input modes:

1. **File path or URL** -- load an existing `.usd`/`.usda`/`.usdc` file.
2. **Inline USDA string** -- compose runtime-generated USD content.
3. **Stage ID** -- reference an existing USD runtime stage.

Multiple USD inputs can be composed together using `path_prefix` to place them at different locations in the stage hierarchy.

## Python

### Load from file or URL

> **Source:** `examples/python/minimal/main.py` snippet `add-usd`

### Load with a path prefix

> **Source:** `tests/test_ovrtx.py` snippet `add-usd-layer`

### Inject inline USDA content

Useful for creating RenderProducts, cameras, or runtime geometry without editing the original scene:

> **Source:** `tests/test_ovrtx.py` snippet `add-usd-layer`

### Compose multiple inputs

> **Source:** `tests/test_ovrtx.py` snippet `add-usd-layer`
>
> See also the planet-system example for composing multiple inputs.

### Remove USD

> **Source:** `tests/test_ovrtx.py` snippet `remove-usd`

## C

### Load from file or URL

> **Source:** `examples/c/minimal/main.cpp` snippet `load-usd-and-wait`

### Poll for completion

Loading is asynchronous in C. Poll until done:

> **Source:** `examples/c/minimal/main.cpp` snippet `load-usd-and-wait`

Or block indefinitely:

> **Source:** `examples/c/minimal/main.cpp` snippet `step-renderer`
>
> The step snippet demonstrates blocking with `ovrtx_timeout_infinite`.

### Load with a path prefix

> **Source:** `examples/c/minimal/main.cpp` snippet `load-usd-and-wait`
>
> Pass a non-empty path prefix as the third argument to `ovrtx_add_usd`.

### Inject inline USDA content

> **Source:** `examples/c/minimal/main.cpp` snippet `load-usd-and-wait`
>
> Set `usd_input.usd_layer_content` instead of `usd_input.usd_file_path` for inline USDA content.

### Remove USD

> **Source:** `tests/test_ovrtx.py` snippet `remove-usd` (Python equivalent)
>
> C: `ovrtx_enqueue_result_t result = ovrtx_remove_usd(renderer, usd_handle);`

### Update time-sampled attributes

For animated USD scenes, update all time-sampled attributes to a specific USD time:

> **Source:** `tests/test_ovrtx.py` snippet `update-from-usd-time`

### Reset stage to empty

Clear all USD content from the runtime stage:

> **Source:** `tests/test_ovrtx.py` snippet `reset-stage`

## Key Types / Functions

| Python | C |
|--------|---|
| `renderer.add_usd(path, prefix)` | `ovrtx_add_usd(renderer, input, prefix, &handle)` |
| `renderer.add_usd_layer(usda, prefix)` | same, but set `usd_input.usd_layer_content` |
| `renderer.remove_usd(handle)` | `ovrtx_remove_usd(renderer, handle)` |
| `renderer.add_usd_async(...)` | `ovrtx_add_usd()` returns immediately; poll with `ovrtx_wait_op` |
| `renderer.update_from_usd_time(usd_time)` | `ovrtx_update_stage_from_usd_time(renderer, usd_time)` |
| `renderer.reset_stage()` / `reset_stage_async()` | `ovrtx_reset_stage(renderer)` |

C input struct (`ovrtx_usd_input_t`) -- set exactly one field:
- `usd_file_path` -- file path or URL
- `usd_stage_id` -- existing runtime stage ID
- `usd_layer_content` -- inline USDA string

## Common Pitfalls

- In the `ovrtx_usd_input_t` struct, set **exactly one** field. Setting multiple is undefined.
- `path_prefix` must not collide with existing prim paths in the stage.
- Inline USDA layers must set `defaultPrim` for reference composition to work.
- In Python, `add_usd()` blocks until loaded. Use `add_usd_async()` if you need non-blocking behavior.
- In C, `ovrtx_add_usd()` is always asynchronous -- you must poll or wait.
- Load errors (e.g., file not found) are reported through `ovrtx_op_wait_result_t::error_op_ids`, not the immediate return value.

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

```python
renderer.add_usd("https://example.com/scene.usda")
# or a local file
renderer.add_usd("/path/to/scene.usda")
```

### Load with a path prefix

```python
renderer.add_usd("/path/to/robot.usda", path_prefix="/World/Robot")
```

### Inject inline USDA content

Useful for creating RenderProducts, cameras, or runtime geometry without editing the original scene:

```python
renderer.add_usd_layer('''
#usda 1.0
(defaultPrim = "Render")
def Scope "Render" {
    def RenderProduct "Camera" {
        rel camera = </World/Camera>
    }
}
''', path_prefix="/Render")
```

### Compose multiple inputs

```python
# Load base scene
renderer.add_usd("scene.usda")

# Add planets under /World/Cube/Orbit
orbit_usda = generate_orbit_layer_usda(num_planets=36)
renderer.add_usd_layer(orbit_usda, path_prefix="/World/Cube/Orbit")
```

### Remove USD

```python
handle = renderer.add_usd("scene.usda")
# later...
renderer.remove_usd(handle)
```

## C

### Load from file or URL

```c
ovrtx_usd_handle_t usd_handle = 0;
ovrtx_usd_input_t usd_input = {};
char const* url = "https://example.com/scene.usda";
usd_input.usd_file_path = {url, strlen(url)};

ovrtx_enqueue_result_t enqueue_result =
    ovrtx_add_usd(renderer, usd_input, {"", 0}, &usd_handle);

if (enqueue_result.status == OVRTX_API_ERROR) {
    ovx_string_t error = ovrtx_get_last_error();
    fprintf(stderr, "add_usd enqueue failed: %.*s\n", (int)error.length, error.ptr);
}
```

### Poll for completion

Loading is asynchronous in C. Poll until done:

```c
ovrtx_op_wait_result_t wait_result;
ovrtx_result_t result = ovrtx_wait_op(
    renderer, enqueue_result.op_index, ovrtx_timeout_t{0}, &wait_result);
THROW_ON_ERROR(result, "wait_op");

// Poll with zero timeout -- returns OVRTX_API_TIMEOUT while still loading
while (ovrtx_wait_op(renderer, enqueue_result.op_index, ovrtx_timeout_t{0},
                     &wait_result).status == OVRTX_API_TIMEOUT) {
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
}
```

Or block indefinitely:

```c
ovrtx_op_wait_result_t wait_result;
ovrtx_result_t result = ovrtx_wait_op(
    renderer, enqueue_result.op_index, ovrtx_timeout_infinite, &wait_result);

if (result.status == OVRTX_API_ERROR) {
    ovx_string_t error = ovrtx_get_last_error();
    fprintf(stderr, "wait_op failed: %.*s\n", (int)error.length, error.ptr);
}
```

### Load with a path prefix

```c
ovrtx_enqueue_result_t enqueue_result =
    ovrtx_add_usd(renderer, usd_input, {"/World/Robot", strlen("/World/Robot")}, &usd_handle);
```

### Inject inline USDA content

```c
ovrtx_usd_input_t layer_input = {};
char const* usda =
    "#usda 1.0\n"
    "(defaultPrim = \"Render\")\n"
    "def Scope \"Render\" {\n"
    "    def RenderProduct \"Camera\" {\n"
    "        rel camera = </World/Camera>\n"
    "    }\n"
    "}\n";
layer_input.usd_layer_content = {usda, strlen(usda)};

ovrtx_usd_handle_t layer_handle = 0;
ovrtx_enqueue_result_t result =
    ovrtx_add_usd(renderer, layer_input, {"/Render", strlen("/Render")}, &layer_handle);
```

### Remove USD

```c
ovrtx_enqueue_result_t result = ovrtx_remove_usd(renderer, usd_handle);
```

## Key Types / Functions

| Python | C |
|--------|---|
| `renderer.add_usd(path, prefix)` | `ovrtx_add_usd(renderer, input, prefix, &handle)` |
| `renderer.add_usd_layer(usda, prefix)` | same, but set `usd_input.usd_layer_content` |
| `renderer.remove_usd(handle)` | `ovrtx_remove_usd(renderer, handle)` |
| `renderer.add_usd_async(...)` | `ovrtx_add_usd()` returns immediately; poll with `ovrtx_wait_op` |

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

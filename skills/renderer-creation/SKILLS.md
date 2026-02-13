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
name: renderer-creation
description: Creating and configuring an ovrtx renderer instance. Use when user asks to create a renderer, initialize ovrtx, configure renderer options, or set up ovrtx.
---

# Renderer Creation

## Overview

The renderer is the central object in ovrtx. You must create one before loading USD scenes, stepping, or reading output. The renderer manages all GPU resources and the internal rendering pipeline.

## Python

Create a renderer with default settings:

```python
import ovrtx

renderer = ovrtx.Renderer()
```

Or with configuration:

```python
from ovrtx import Renderer, RendererConfig

config = RendererConfig(
    sync_mode=True,           # Synchronous mode (blocks on enqueue, useful for debugging)
    log_file_path="/tmp/ovrtx.log",  # Log file path (supports ${start_timestamp} token)
    log_level="info",         # "verbose", "info", "warn", "error"
)
renderer = Renderer(config=config)
```

Cleanup is automatic -- the renderer is destroyed when it goes out of scope.

## C

Create a renderer with no configuration:

```c
#include <ovrtx/ovrtx.h>
#include <ovrtx/ovrtx_config.h>

ovrtx_renderer_t* renderer = NULL;
ovrtx_config_t config = {};
ovrtx_result_t result = ovrtx_create_renderer(&config, &renderer);
if (result.status == OVRTX_API_ERROR) {
    ovx_string_t error = ovrtx_get_last_error();
    // handle error...
}
```

### Optional explicit initialize/shutdown

`ovrtx_initialize()` is optional. If you do not call it, `ovrtx_create_renderer()` initializes ovrtx automatically.

Use explicit initialize/shutdown when creating and destroying multiple renderers in a process and you want one-time loader initialization to happen once:

```c
ovrtx_config_t init_config = {};
ovrtx_result_t result = ovrtx_initialize(&init_config);
if (result.status == OVRTX_API_ERROR) {
    // handle init failure...
}

// create/destroy one or more renderers...

ovrtx_shutdown();
```

With configuration entries:

```c
ovrtx_renderer_config_entry_t entries[] = {
    ovrtx_config_entry_sync_mode(true),
    ovrtx_config_entry_log_level({"info", 4}),
    ovrtx_config_entry_binary_package_root_path({"/path/to/ovrtx", strlen("/path/to/ovrtx")}),
};

ovrtx_config_t config;
config.entries = entries;
config.entry_count = sizeof(entries) / sizeof(entries[0]);

ovrtx_renderer_t* renderer = NULL;
ovrtx_result_t result = ovrtx_create_renderer(&config, &renderer);
```

Cleanup is manual -- you must call `ovrtx_destroy_renderer()`:

```c
ovrtx_destroy_renderer(renderer);
```

## Key Types / Functions

| Python | C |
|--------|---|
| `Renderer(config=None)` | `ovrtx_create_renderer(config, &renderer)` |
| `RendererConfig` | `ovrtx_config_t` + `ovrtx_renderer_config_entry_t` |
| automatic `__del__` | `ovrtx_destroy_renderer(renderer)` |

Config helpers in C (`ovrtx_config.h`):
- `ovrtx_config_entry_sync_mode(bool)`
- `ovrtx_config_entry_log_file_path(ovx_string_t)`
- `ovrtx_config_entry_log_level(ovx_string_t)`
- `ovrtx_config_entry_binary_package_root_path(ovx_string_t)`
- `ovrtx_config_entry_enable_profiling(bool)`
- `ovrtx_config_entry_read_gpu_transforms(bool)`
- `ovrtx_config_entry_output_partial_frames(bool)`

## Common Pitfalls

- In C, forgetting to call `ovrtx_destroy_renderer()` will leak GPU resources.
- `binary_package_root_path` is only required when static linking ovrtx, or when your install layout breaks apart the ovrtx `bin/` directory.
- With dynamic linking, ovrtx expects runtime directories under `bin/` (`cache/`, `library/`, `libs/`, `mdl/`, `plugins/`, `rendering-data/`, `usd_plugins/`) to be found next to `ovrtx-dynamic.dll` / `libovrtx-dynamic.so`.
- If you call `ovrtx_initialize()` explicitly, pair it with a matching `ovrtx_shutdown()`.
- Error strings from `ovrtx_get_last_error()` are only valid until the next API call on the same thread.

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

> **Source:** `examples/python/minimal/main.py` snippet `create-renderer`

Or with configuration:

> **Source:** `tests/conftest.py` snippet `renderer-config`

Cleanup is automatic -- the renderer is destroyed when it goes out of scope.

## C

Create a renderer with no configuration:

> **Source:** `examples/c/minimal/main.cpp` snippet `create-renderer`

### Optional explicit initialize/shutdown

`ovrtx_initialize()` is optional. If you do not call it, `ovrtx_create_renderer()` initializes ovrtx automatically.

Use explicit initialize/shutdown when creating and destroying multiple renderers in a process and you want one-time loader initialization to happen once:

> **Source:** `examples/c/vulkan-interop/src/main.cpp` snippet `initialize-and-create-renderer`

With configuration entries:

> **Source:** `examples/c/minimal/main.cpp` snippet `create-renderer`
>
> To add config entries, populate `config.entries` and `config.entry_count` before calling `ovrtx_create_renderer`. See the `ovrtx_config_entry_*` helpers in `ovrtx_config.h`.

Cleanup is manual -- you must call `ovrtx_destroy_renderer()`:

> **Source:** `examples/c/minimal/main.cpp` snippet `unmap-and-cleanup`

## Key Types / Functions

| Python | C |
|--------|---|
| `Renderer(config=None)` | `ovrtx_create_renderer(config, &renderer)` |
| `RendererConfig` | `ovrtx_config_t` + `ovrtx_config_entry_t` |
| automatic `__del__` | `ovrtx_destroy_renderer(renderer)` |
| `renderer.version` → `(major, minor, patch)` | `ovrtx_get_version(&major, &minor, &patch)` (callable before init) |
| `renderer.config` → `RendererConfig` | (no C equivalent -- config was passed at creation) |

Config helpers in C (`ovrtx_config.h`):
- `ovrtx_config_entry_sync_mode(bool)`
- `ovrtx_config_entry_log_file_path(ovx_string_t)`
- `ovrtx_config_entry_log_level(ovx_string_t)`
- `ovrtx_config_entry_binary_package_root_path(ovx_string_t)`
- `ovrtx_config_entry_enable_profiling(bool)`
- `ovrtx_config_entry_read_gpu_transforms(bool)`
- `ovrtx_config_entry_output_partial_frames(bool)`
- `ovrtx_config_entry_keep_system_alive(bool)` -- keep shared GPU resources alive after the last renderer is destroyed, so a subsequent `ovrtx_create_renderer` reuses them
- `ovrtx_config_entry_active_cuda_gpus(ovx_string_t)` -- comma-separated CUDA device indices (e.g., `"0,1,2"`) to select which GPUs to use for rendering

## Common Pitfalls

- In C, forgetting to call `ovrtx_destroy_renderer()` will leak GPU resources.
- `binary_package_root_path` is only required when static linking ovrtx, or when your install layout breaks apart the ovrtx `bin/` directory.
- With dynamic linking, ovrtx expects runtime directories under `bin/` (`cache/`, `library/`, `libs/`, `mdl/`, `plugins/`, `rendering-data/`, `usd_plugins/`) to be found next to `ovrtx-dynamic.dll` / `libovrtx-dynamic.so`.
- If you call `ovrtx_initialize()` explicitly, pair it with a matching `ovrtx_shutdown()`.
- Error strings from `ovrtx_get_last_error()` are only valid until the next API call on the same thread.

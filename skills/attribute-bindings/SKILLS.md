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
name: attribute-bindings
description: Creating persistent attribute bindings for efficient repeated writes. Use when user asks about persistent bindings, repeated writes, efficient animation loops, or bind_attribute.
---

# Attribute Bindings

## Overview

When writing the same attribute to the same set of prims every frame (e.g., animating transforms), creating a persistent binding avoids the overhead of rebuilding the binding descriptor each time. Create the binding once, then call `write()` or `map()` on it repeatedly.

## Python

### Direct write (no binding)

For one-shot or infrequent writes, `renderer.write_attribute()` rebuilds the binding descriptor each call:

> **Source:** `tests/test_ovrtx.py` snippet `write-attribute-cpu`

### Create binding and write

For repeated writes to the same attribute/prims, create a persistent binding once:

> **Source:** `tests/test_ovrtx.py` snippet `bind-attribute-write`

### Reuse binding for repeated writes

> **Source:** `tests/test_ovrtx.py` snippet `bind-attribute-multiple-writes`

### Array attribute binding

> **Source:** `tests/test_ovrtx.py` snippet `bind-array-attribute`

### Binding with explicit semantic

> **Source:** `tests/test_ovrtx.py` snippet `bind-write-semantic`

### Use binding for mapping (zero-copy)

> **Source:** `tests/test_ovrtx.py` snippet `map-binding-cpu`

## C

### Create and use a persistent binding

> **Source:** `examples/c/vulkan-interop/src/main.cpp` snippet `write-camera-transform`

## Key Types / Functions

| Python | C |
|--------|---|
| `renderer.bind_attribute(...)` | `ovrtx_create_attribute_binding(renderer, &desc, &handle)` |
| `renderer.bind_array_attribute(...)` | same, with `is_array=true` in desc |
| `binding.write(data, data_access=DataAccess.SYNC, cuda_stream=, cuda_event=)` | `ovrtx_write_attribute(renderer, &binding_ref, &buffer, access)` |
| `binding.map(device=...)` | `ovrtx_map_attribute(renderer, &binding_ref, mapping_desc, &out)` |
| `binding.unbind()` | `ovrtx_destroy_attribute_binding(renderer, handle)` |

Binding flags (C only, set via `desc.flags`):
- `OVRTX_BINDING_FLAG_NONE` -- default
- `OVRTX_BINDING_FLAG_OPTIMIZE` -- optimize internal structures for frequent high-volume writes

## Common Pitfalls

- The canonical transform attribute name is `"omni:xform"`. The legacy name `"omni:fabric:localMatrix"` (used in examples above) is also accepted. New code should prefer `"omni:xform"`.
- Always call `unbind()` (Python) or `ovrtx_destroy_attribute_binding()` (C) when done. In Python, the binding auto-unbinds on garbage collection, but explicit cleanup is preferred.
- The binding locks in the prim list, attribute name, and element type. You cannot change these after creation.
- In C, `OVRTX_BINDING_FLAG_OPTIMIZE` should be used for the primary hot-path binding. The last binding created with this flag takes priority.
- In Python, `bind_attribute` is synchronous (blocks until the binding is ready). Use `bind_attribute_async` for non-blocking creation.

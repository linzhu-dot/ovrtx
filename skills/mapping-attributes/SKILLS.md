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
name: mapping-attributes
description: Zero-copy attribute map/unmap for direct memory access to RTX internal buffers. Use when user asks about zero-copy writes, map attribute, direct memory access, Warp kernel writes, or GPU attribute updates.
---

# Mapping Attributes

## Overview

Mapping gives you direct access to RTX's internal Fabric buffer for an attribute. Instead of copying data in (like `write_attribute`), you write directly into the buffer, then unmap to apply changes. This is the most efficient path for per-frame updates, especially with GPU compute (e.g., Warp kernels).

The pattern is: **map -> write into tensor -> unmap**.

## Python

### CPU mapping with NumPy

> **Source:** `tests/test_ovrtx.py` snippet `map-attribute-cpu`

### Context manager (recommended)

> **Source:** `tests/test_ovrtx.py` snippet `map-attribute-cpu`

### GPU mapping with Warp

From the planet-system example -- map on CUDA, compute with a Warp kernel, unmap with stream sync:

> **Source:** `tests/test_ovrtx.py` snippet `unmap-attribute-cuda-sync`

### Using a persistent binding for mapping

More efficient for repeated map/unmap cycles (avoids recreating the binding descriptor):

> **Source:** `tests/test_ovrtx.py` snippet `map-binding-cpu`

## C

### Map, write, unmap

> **Source:** `examples/c/vulkan-interop/src/main.cpp` snippet `write-camera-transform`

### GPU mapping with CUDA sync

> **Source:** `examples/c/vulkan-interop/src/main.cpp` snippet `map-rendered-output-cuda-array`

## Key Types / Functions

| Python | C |
|--------|---|
| `renderer.map_attribute(...)` | `ovrtx_map_attribute(renderer, &binding, desc, &out)` |
| `renderer.unmap_attribute(mapping)` | `ovrtx_unmap_attribute(renderer, handle, sync)` |
| `binding.map(device=...)` | same, with binding handle |
| `mapping.tensor` | `out_mapping.dl` (DLTensor) |
| `mapping.unmap(stream=...)` | `ovrtx_unmap_attribute` with `cuda_sync.stream` |
| `mapping.unmap(event=...)` | `ovrtx_unmap_attribute` with `cuda_sync.wait_event` |
| `renderer.unmap_attribute_async(mapping, ...)` | `ovrtx_unmap_attribute` (always async in C) |

Semantic-aware tensor shapes in Python:
- When using `dtype`/`shape` (preferred for interop), the tensor has shape `(N, *shape)` — e.g. `(N, 4, 4)` for `dtype="float64", shape=(4, 4)`.
- The `semantic=` path also reshapes automatically: `Semantic.XFORM_MAT4x4` → `(N, 4, 4)`, points → `(N, 3)`, colors → `(N, 4)` for RGBA or `(N, 3)` for RGB.

## Common Pitfalls

- The canonical transform attribute name is `"omni:xform"`. The legacy name `"omni:fabric:localMatrix"` (used in examples above) is also accepted. New code should prefer `"omni:xform"`.
- Array attributes (e.g., `float3[] points`) are **not supported** for mapping because they have variable lengths per prim. Use `write_array_attribute()` instead.
- Data must be fully written before calling `unmap()`. For CUDA, pass `stream` or `event` so ovrtx knows when the GPU write is complete.
- `event` and `stream` are mutually exclusive on unmap -- use one or the other.
- Providing `event`/`stream` for a CPU-mapped attribute raises an error.
- Multiple mappings can be outstanding on the same attribute. The logical order of application depends on the order of `unmap()` calls.

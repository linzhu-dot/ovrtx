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
name: writing-transforms
description: Writing 4x4 transform matrices to move, rotate, or scale prims. Use when user asks to move an object, set a transform, update position, animate a transform, or set a camera transform.
---

# Writing Transforms

## Overview

Transforms control the position, rotation, and scale of prims in the scene. ovrtx supports three transform representations. The most common is a 4x4 matrix of doubles using the USD row-vector convention: translation is in the last row (`[3][0..2]`). The canonical transform attribute name is `"omni:xform"` (used by the C convenience helpers in `ovrtx_attributes.h`). The legacy name `"omni:fabric:localMatrix"` is also accepted.

## Python

### Write a 4x4 transform (single prim)

> **Source:** `tests/test_ovrtx.py` snippet `write-attribute-cpu`

### Write transforms for multiple prims

> **Source:** `tests/test_ovrtx.py` snippet `bind-attribute-write`

## C

### Write identity transform on a single prim

> **Source:** `examples/c/vulkan-interop/src/main.cpp` snippet `write-camera-transform`

### Batch write (N prims)

> **Source:** `examples/c/vulkan-interop/src/main.cpp` snippet `write-camera-transform`
>
> For multiple prims, increase `prim_list.num_paths` and provide a contiguous array of transforms.

### C convenience helpers (`ovrtx_attributes.h`)

Instead of building DLTensor and binding descriptors manually, use the convenience helpers that handle the boilerplate:

> **Source:** No example coverage yet. C convenience API:
>
> `#include <ovrtx/ovrtx_attributes.h>` provides `ovrtx_set_xform_mat(renderer, paths, count, transforms)`, `ovrtx_set_xform_pos_rot_scale(...)`, `ovrtx_set_xform_pos_rot3x3(...)`, and `ovrtx_set_reset_xform_stack(...)`. All write to `"omni:xform"` and return `ovrtx_enqueue_result_t`.

## Key Types / Functions

| Representation | Python | C | Size per prim |
|----------------|--------|---|---------------|
| 4x4 matrix (doubles) | `Semantic.XFORM_MAT4x4` or `dtype="float64", shape=(4, 4)` | `OVRTX_SEMANTIC_XFORM_MAT4x4` / `{ kDLFloat, 64, 16 }` | 128 bytes |
| pos3d + rot4f + scale3f | (C only) | `OVRTX_SEMANTIC_XFORM_POS3d_ROT4f_SCALE3f` | 56 bytes |
| pos3d + rot3x3f | (C only) | `OVRTX_SEMANTIC_XFORM_POS3d_ROT3x3f` | 64 bytes |

C convenience helpers (`#include <ovrtx/ovrtx_attributes.h>`):
- `ovrtx_set_xform_mat(renderer, paths, count, transforms)` -- 4x4 matrix
- `ovrtx_set_xform_pos_rot_scale(renderer, paths, count, transforms)` -- pos + quat + scale
- `ovrtx_set_xform_pos_rot3x3(renderer, paths, count, transforms)` -- pos + 3x3 rotation
- `ovrtx_set_reset_xform_stack(renderer, paths, count, values)` -- set `omni:resetXformStack`

Python semantic: `Semantic.XFORM_MAT4x4` (from `from ovrtx import Semantic`). For interop, the preferred approach is `dtype="float64", shape=(4, 4)` on `bind_attribute`/`map_attribute`, or passing an `(N, 4, 4)` float64 array directly to `write_attribute` (type is auto-inferred when the attribute already exists in the stage). The `dtype` parameter accepts string names (`"float64"`), NumPy dtypes (`np.float64`), or Python built-ins (`float`). The `semantic=` parameter is required when creating attributes from scratch via `prim_mode=PrimMode.CREATE_NEW`.

## Common Pitfalls

- Matrices use the **USD row-vector convention** (same as `GfMatrix4d`). Translation is in the last row: `v[12]`, `v[13]`, `v[14]` (or `matrix[3][0..2]` in Python/C).
- The canonical transform attribute name is `"omni:xform"`. The legacy name `"omni:fabric:localMatrix"` is also accepted. New code should prefer `"omni:xform"` to match the C convenience helpers (`ovrtx_set_xform_mat`, etc.).
- Transform dtype is `float64` (doubles), not float32. Using the wrong dtype will cause errors.
- For repeated per-frame transform updates, use attribute bindings or mapping for better performance (see `attribute-bindings` and `mapping-attributes` skills).
- Earlier code may use `Matrix4d.to_dltensor()` for single-prim writes. This still works, but the preferred approach is to use `dtype="float64", shape=(4, 4)` with `bind_attribute`/`map_attribute`, or pass an `(N, 4, 4)` float64 array directly to `write_attribute`.

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
name: writing-attributes
description: Writing scalar and array attribute data to prims. Use when user asks to write an attribute, set a property, change a material, set a color, or modify mesh data.
---

# Writing Attributes

## Overview

Beyond transforms, you can write arbitrary attributes to prims: colors, visibility, mesh geometry, string tokens, and more. ovrtx uses DLTensor as the data format, and the dtype must exactly match the USD attribute schema.

There are two categories:
- **Scalar attributes** -- one value per prim (e.g., a color, a transform)
- **Array attributes** -- variable-length per prim (e.g., mesh points, face vertex counts)

## Python

### Array attribute write (mesh points)

> **Source:** `tests/test_ovrtx.py` snippet `write-array-attribute-points`

### Array attribute write (face vertex counts)

> **Source:** `tests/test_ovrtx.py` snippet `write-array-attribute`

### Prim modes

> **Source:** `tests/test_ovrtx.py` snippet `bind-attribute-write`

### Token string attribute

> **Source:** `tests/test_ovrtx.py` snippet `write-attribute-token-string`

### Path/relationship array attribute

> **Source:** `tests/test_ovrtx.py` snippet `write-array-attribute-path-string`

## C

### Generic scalar write in C

> **Source:** `examples/c/vulkan-interop/src/main.cpp` snippet `write-camera-transform`

## Key Types / Functions

| Python | C |
|--------|---|
| `renderer.write_attribute(...)` | `ovrtx_write_attribute(renderer, &binding, &buffer, access)` → returns `ovrtx_enqueue_result_t` |
| `renderer.write_array_attribute(..., is_token=True)` | same, with `is_array=true` in binding desc |

Semantics (Python: `from ovrtx import Semantic`; C: `ovrtx_attribute_semantic_t`):
- `Semantic.NONE` / `OVRTX_SEMANTIC_NONE` -- generic data; type is determined by `DLDataType` (dtype and lanes). Use for colors, visibility, and other non-structured attributes.
- `Semantic.XFORM_MAT4x4` / `OVRTX_SEMANTIC_XFORM_MAT4x4` -- 4x4 double matrix
- `OVRTX_SEMANTIC_XFORM_POS3d_ROT4f_SCALE3f` -- decomposed transform (C only)
- `OVRTX_SEMANTIC_XFORM_POS3d_ROT3x3f` -- decomposed transform (C only)
- `Semantic.PATH_STRING` / `OVRTX_SEMANTIC_PATH_STRING` -- prim path strings
- `Semantic.TOKEN_STRING` / `OVRTX_SEMANTIC_TOKEN_STRING` -- token strings

In C, `ovrtx_write_attribute` returns `ovrtx_enqueue_result_t` which contains both `.status` (check for `OVRTX_API_ERROR`) and `.op_index` (for async tracking via `ovrtx_wait_op`).

Data access modes (Python: `from ovrtx import DataAccess`):
- `DataAccess.SYNC` -- copies data during the call, safe to free after return
- `DataAccess.ASYNC` -- data accessed later during stream execution, must keep alive; pass `cuda_stream=` or `cuda_event=` for GPU synchronization. Not allowed with string data.

## Common Pitfalls

- Array attribute dtype must exactly match the USD schema. Using numpy's default `float64` for a `float3[]` attribute (which expects `float32`) will cause errors.
- String attributes (`Semantic.PATH_STRING`, `Semantic.TOKEN_STRING`) only support synchronous data access (`DataAccess.SYNC`).
- For array attributes in Python, pass a list of tensors (one per prim), not a single tensor. NumPy arrays, Warp arrays, and any `__dlpack__`-compatible objects are accepted directly.
- When creating attributes from scratch (`prim_mode=PrimMode.CREATE_NEW`), `semantic=` is required for structured types (transforms, packed structs) so the C library knows what USD attribute type to create.
- For `write_array_attribute` with `list[list[str]]` data, strings default to path/relationship arrays. Set `is_token=True` to write token arrays (e.g., `xformOpOrder`, `apiSchemas`).
- `dirty_bits` is a bitvector with 1 bit per prim -- the byte array size must be `(prim_count + 7) / 8`.
- In C, `dirty_bits` support three combination modes via `ovrtx_write_bits_t` in the `ovrtx_input_buffer_t.dirty_bits_mode` field: `OVRTX_DIRTY_MASK_REPLACE` (default -- replace existing mask), `OVRTX_DIRTY_MASK_OR` (merge with existing), `OVRTX_DIRTY_MASK_AND` (intersect with existing).

C convenience helpers for string attributes (`#include <ovrtx/ovrtx_attributes.h>`):
- `ovrtx_set_path_attributes(renderer, paths, count, attr_name, path_values)` -- write path/relationship attributes. Each prim gets a single-element array (relationships are always arrays in USD).
- `ovrtx_set_token_attributes(renderer, paths, count, attr_name, token_values)` -- write token string attributes (one per prim).

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
name: cloning-prims
description: Cloning USD subtrees to create copies at new paths. Use when user asks to clone, duplicate, copy a prim, or create instances of existing geometry.
---

# Cloning Prims

## Overview

`clone_usd` creates copies of an existing prim subtree at one or more new target paths in the runtime stage. This is useful for duplicating geometry, creating arrays of objects, or spawning instances from a template.

## Python

### Clone to a single target

> **Source:** `tests/test_ovrtx.py` snippet `clone-usd`

### Clone to multiple targets

> **Source:** `tests/test_ovrtx.py` snippet `clone-usd`

### Async clone

> **Source:** `tests/test_ovrtx.py` snippet `clone-usd`
>
> For async: `op = renderer.clone_usd_async(source, targets)` followed by `op.wait()`.

## C

### Clone to multiple targets

> **Source:** `tests/test_ovrtx.py` snippet `clone-usd` (Python equivalent)
>
> C: `ovrtx_enqueue_result_t result = ovrtx_clone_usd(renderer, source, targets, count);`

### Wait for completion

> **Source:** `examples/c/minimal/main.cpp` snippet `load-usd-and-wait`
>
> Same `ovrtx_wait_op` pattern applies to clone operations.

## Key Types / Functions

| Python | C |
|--------|---|
| `renderer.clone_usd(source, targets)` | `ovrtx_clone_usd(renderer, source, targets, count)` |
| `renderer.clone_usd_async(source, targets)` | `ovrtx_clone_usd()` (always async in C) |

## Common Pitfalls

- The source path **must exist** in the stage.
- The target paths **must not already exist** in the stage.
- Cloning copies the entire subtree under the source path, including all children.
- In Python, `clone_usd()` blocks until the operation completes. Use `clone_usd_async()` for non-blocking behavior.
- In C, `ovrtx_clone_usd()` is always asynchronous. You **must** wait on the returned `op_index` with `ovrtx_wait_op()` before using the cloned prims (e.g., writing attributes or stepping).

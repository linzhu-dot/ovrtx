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

```python
renderer.clone_usd("/World/Template", ["/World/Copy1"])
```

### Clone to multiple targets

```python
renderer.clone_usd(
    "/World/Robot",
    [f"/World/Robot_{i}" for i in range(10)]
)
```

### Async clone

```python
op = renderer.clone_usd_async("/World/Source", ["/World/Target"])
op.wait()  # or poll with timeout
```

## C

### Clone to multiple targets

```c
ovx_string_t source = {"/World/Template", strlen("/World/Template")};

ovx_string_t targets[] = {
    {"/World/Copy_0", strlen("/World/Copy_0")},
    {"/World/Copy_1", strlen("/World/Copy_1")},
    {"/World/Copy_2", strlen("/World/Copy_2")},
};

ovrtx_enqueue_result_t result = ovrtx_clone_usd(
    renderer, source, targets, 3);

if (result.status == OVRTX_API_ERROR) {
    ovx_string_t error = ovrtx_get_last_error();
    fprintf(stderr, "clone_usd enqueue failed: %.*s\n",
            (int)error.length, error.ptr);
}
```

### Wait for completion

```c
ovrtx_op_wait_result_t wait_result;
ovrtx_result_t wait_status =
    ovrtx_wait_op(renderer, result.op_index, ovrtx_timeout_infinite, &wait_result);

if (wait_status.status == OVRTX_API_ERROR) {
    ovx_string_t error = ovrtx_get_last_error();
    fprintf(stderr, "wait_op failed: %.*s\n", (int)error.length, error.ptr);
}

if (wait_result.num_error_ops > 0) {
    for (size_t i = 0; i < wait_result.num_error_ops; ++i) {
        ovrtx_op_id_t failed_id = wait_result.error_op_ids[i];
        ovx_string_t error = ovrtx_get_last_op_error(failed_id);
        fprintf(stderr, "Clone op %llu failed: %.*s\n",
                (unsigned long long)failed_id,
                (int)error.length, error.ptr);
    }
}
```

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

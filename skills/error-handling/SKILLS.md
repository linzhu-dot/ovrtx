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
name: error-handling
description: Error checking patterns for both C and Python. Use when user asks about error handling, checking errors, debugging ovrtx failures, or troubleshooting.
---

# Error Handling

## Overview

In Python, ovrtx raises `RuntimeError` with descriptive messages on failure. In C, every API call returns a status code that must be checked, and error details are retrieved with `ovrtx_get_last_error()`.

## Python

### Automatic error handling

All Python API methods raise `RuntimeError` on failure:

```python
try:
    renderer = ovrtx.Renderer()
    renderer.add_usd("/nonexistent/file.usda")
except RuntimeError as e:
    print(f"Error: {e}")
```

### Async operation errors

Errors from async operations surface when you call `wait()`:

```python
op = renderer.add_usd_async("/nonexistent/file.usda")
try:
    op.wait()
except RuntimeError as e:
    print(f"Load failed: {e}")
```

### Step errors

```python
try:
    products = renderer.step(
        render_products={"/Render/Camera"},
        delta_time=1.0/60
    )
except RuntimeError as e:
    print(f"Step failed: {e}")
```

## C

### Check every return value

```c
bool initialized = false;
ovrtx_renderer_t* renderer = nullptr;
ovrtx_step_result_handle_t step_result = OVRTX_INVALID_HANDLE;

ovrtx_result_t result = ovrtx_initialize(&config);
if (result.status == OVRTX_API_ERROR) {
    ovx_string_t error = ovrtx_get_last_error();
    fprintf(stderr, "initialize failed: %.*s\n", (int)error.length, error.ptr);
    goto cleanup;
}
initialized = true;

result = ovrtx_create_renderer(&config, &renderer);
if (result.status == OVRTX_API_ERROR) {
    ovx_string_t error = ovrtx_get_last_error();
    fprintf(stderr, "create_renderer failed: %.*s\n", (int)error.length, error.ptr);
    goto cleanup;
}

// ... do work, possibly creating step_result ...

cleanup:
if (step_result != OVRTX_INVALID_HANDLE && renderer) {
    ovrtx_destroy_results(renderer, step_result);
}
if (renderer) {
    ovrtx_destroy_renderer(renderer);
}
if (initialized) {
    ovrtx_shutdown();
}
```

### THROW_ON_ERROR macro pattern

From the minimal C example -- wraps the check-and-throw pattern:

```c
#define THROW_ON_ERROR(RESULT, OPERATION)                                      \
    do {                                                                       \
        if (RESULT.status == OVRTX_API_ERROR) {                                \
            ovx_string_t error = ovrtx_get_last_error();                       \
            char error_msg[512];                                               \
            if (error.ptr && error.length > 0) {                               \
                snprintf(error_msg, sizeof(error_msg),                         \
                         "ovrtx %s failed: %.*s",                              \
                         OPERATION,                                            \
                         (int)error.length, error.ptr);                        \
            } else {                                                           \
                snprintf(error_msg, sizeof(error_msg),                         \
                         "ovrtx %s failed", OPERATION);                        \
            }                                                                  \
            throw std::runtime_error(error_msg);                               \
        }                                                                      \
    } while (0)

// Usage:
ovrtx_result_t result = ovrtx_create_renderer(&config, &renderer);
THROW_ON_ERROR(result, "create_renderer");
```

### Per-operation errors from wait

Asynchronous errors (e.g., file not found during USD load) are reported through `ovrtx_wait_op`:

```c
ovrtx_op_wait_result_t wait_result;
ovrtx_result_t result = ovrtx_wait_op(
    renderer, op_id, ovrtx_timeout_infinite, &wait_result);

if (result.status == OVRTX_API_ERROR) {
    ovx_string_t error = ovrtx_get_last_error();
    fprintf(stderr, "wait_op failed: %.*s\n", (int)error.length, error.ptr);
}

// Check for per-operation errors
if (wait_result.num_error_ops > 0) {
    for (size_t i = 0; i < wait_result.num_error_ops; ++i) {
        ovrtx_op_id_t failed_id = wait_result.error_op_ids[i];
        ovx_string_t error = ovrtx_get_last_op_error(failed_id);
        fprintf(stderr, "Operation %llu failed: %.*s\n",
                (unsigned long long)failed_id,
                (int)error.length, error.ptr);
    }
}
```

## Key Types / Functions

| Python | C |
|--------|---|
| `RuntimeError` | `result.status == OVRTX_API_ERROR` |
| exception message | `ovrtx_get_last_error()` |
| async error on `wait()` | `ovrtx_get_last_op_error(op_id)` |

C status codes:
- `OVRTX_API_SUCCESS` (0) -- success
- `OVRTX_API_ERROR` (1) -- error, call `ovrtx_get_last_error()` for details
- `OVRTX_API_TIMEOUT` (2) -- timeout reached

## Common Pitfalls

- Error strings from `ovrtx_get_last_error()` live in thread-local storage and are invalidated by the next API call on the same thread. Copy the string if you need to keep it.
- `ovrtx_get_last_op_error()` strings are valid only until the next `ovrtx_wait_op` call.
- In C error paths, clean up explicitly: destroy step results if alive, destroy renderer if created, and call `ovrtx_shutdown()` if `ovrtx_initialize()` was called.
- In C, some enqueue operations (like `ovrtx_add_usd`) may succeed at enqueue time but fail during async execution. Always check `wait_result.error_op_ids` after waiting.
- In Python, some errors during `__del__` cleanup are printed to stderr rather than raised (since Python does not allow exceptions in destructors).

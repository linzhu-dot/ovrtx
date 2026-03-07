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

> **Source:** `tests/test_ovrtx.py` snippet `step-async-polling`
>
> All Python API methods raise `RuntimeError` on failure. The test demonstrates the async error surface pattern.

### Async operation errors

Errors from async operations surface when you call `wait()`:

> **Source:** `tests/test_ovrtx.py` snippet `step-async-polling`

### Step errors

> **Source:** `examples/python/minimal/main.py` snippet `step`
>
> Wrap in `try/except RuntimeError` to handle step failures.

## C

### Check every return value

> **Source:** `examples/c/minimal/main.cpp` snippet `create-renderer`
>
> Followed by: `examples/c/minimal/main.cpp` snippet `load-usd-and-wait`
>
> Every API call returns a status code. Check with the `check-error-helper` snippet pattern.

### check_and_print_error helper function

From the minimal C example -- a template helper that checks the status, prints to `std::cerr`, and returns whether an error occurred. Works with both `ovrtx_result_t` and `ovrtx_enqueue_result_t`:

> **Source:** `examples/c/minimal/main.cpp` snippet `check-error-helper`

### Per-operation errors from wait

Asynchronous errors (e.g., file not found during USD load) are reported through `ovrtx_wait_op`:

> **Source:** `examples/c/minimal/main.cpp` snippet `load-usd-and-wait`
>
> Check `wait_result.num_error_ops` and iterate `wait_result.error_op_ids` after `ovrtx_wait_op`.

### Log callback (C)

Set a callback to receive log messages from operations, with severity filtering:

> **Source:** No example coverage yet. C API pattern:
>
> Call `ovrtx_set_log_callback(renderer, min_severity, channel, callback, user_data)` to register a log handler. Call `ovrtx_flush_op_log(renderer, timeout)` to flush pending messages. Pass `NULL` as callback to disable. See `ovrtx_log_severity_t` levels: `OVRTX_LOG_INFO` (0), `OVRTX_LOG_WARNING` (1), `OVRTX_LOG_ERROR` (2).

Log severity levels: `OVRTX_LOG_INFO` (0), `OVRTX_LOG_WARNING` (1), `OVRTX_LOG_ERROR` (2). The callback may be invoked from any thread but is serialized per renderer instance. Message strings are only valid during the callback.

## Key Types / Functions

| Python | C |
|--------|---|
| `RuntimeError` | `result.status == OVRTX_API_ERROR` |
| exception message | `ovrtx_get_last_error()` |
| async error on `wait()` | `ovrtx_get_last_op_error(op_id)` |
| (not exposed) | `ovrtx_set_log_callback(renderer, min_severity, channel, callback, user_data)` |
| (not exposed) | `ovrtx_flush_op_log(renderer, timeout)` |

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

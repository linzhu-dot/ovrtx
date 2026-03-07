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
name: async-operations
description: Asynchronous operation patterns including polling, timeouts, and non-blocking workflows. Use when user asks about async rendering, non-blocking operations, polling, timeouts, or parallel rendering.
---

# Async Operations

## Overview

All ovrtx enqueue operations (add_usd, step, write_attribute, etc.) are internally asynchronous and stream-ordered. In Python, the default methods (e.g., `add_usd`, `step`) block until completion. The `_async` variants return `Operation` objects that support polling and custom timeouts.

In C, all enqueue calls return immediately with an `ovrtx_op_id_t` that can be polled or waited on.

## Python

### Non-blocking USD load

> **Source:** `tests/test_ovrtx.py` snippet `step-async-polling`
>
> The same polling pattern applies to `add_usd_async` — call `op.wait(timeout_ns=0)` to poll, or `op.wait()` to block.

### Non-blocking step with two-stage timeout

> **Source:** `tests/test_ovrtx.py` snippet `step-async-polling`

### Infinite wait (default)

> **Source:** `examples/python/minimal/main.py` snippet `step`
>
> The synchronous `step()` is equivalent to `step_async().wait()`.

## C

### Poll with zero timeout

> **Source:** `examples/c/minimal/main.cpp` snippet `load-usd-and-wait`
>
> The load snippet demonstrates polling with `ovrtx_timeout_t{0}`.

### Poll loop with sleep

> **Source:** `examples/c/minimal/main.cpp` snippet `load-usd-and-wait`

### Block indefinitely

> **Source:** `examples/c/minimal/main.cpp` snippet `step-renderer`
>
> Uses `ovrtx_timeout_infinite` to block until completion.

### Check for errors after wait

> **Source:** `examples/c/minimal/main.cpp` snippet `load-usd-and-wait`
>
> Check `wait_result.num_error_ops` after any `ovrtx_wait_op` call.

### Query operation progress (C)

For long-running operations (e.g., USD loading), you can query progress and named resource counters:

> **Source:** No example coverage yet. C API pattern:
>
> Call `ovrtx_query_op_status(renderer, op_id, &status)` to get progress and counters, then `ovrtx_release_op_status(renderer, &status)` to release. See the `ovrtx_op_status_t` type documentation.

Counter names are operation-dependent (e.g., `"shaders"`, `"textures"`, `"materials"` during USD loading). A `total` of 0 means the total is not yet known.

## Key Types / Functions

| Python | C |
|--------|---|
| `Operation.wait(timeout_ns=None)` | `ovrtx_wait_op(renderer, op_id, timeout, &wait_result)` |
| `Operation.wait(timeout_ns=0)` | `ovrtx_wait_op` with `timeout.time_out_ns = 0` |
| `RendererResult.wait(step_timeout_ns, fetch_timeout_ns)` | manual two-stage: `ovrtx_wait_op` then `ovrtx_fetch_results` |
| `RuntimeError` on failure | `wait_result.num_error_ops > 0` |
| (not exposed) | `ovrtx_query_op_status(renderer, op_id, &status)` -- progress + resource counters |
| (not exposed) | `ovrtx_release_op_status(renderer, &status)` -- must call after each query |

C timeout constants:
- `ovrtx_timeout_t{0}` -- non-blocking poll
- `ovrtx_timeout_infinite` -- block indefinitely
- `ovrtx_timeout_t{5000000000}` -- 5 seconds in nanoseconds

C wait result fields:
- `error_op_ids` / `num_error_ops` -- operations that errored since last wait
- `lowest_pending_op_id` -- 0 if all operations complete, nonzero if still pending

## Common Pitfalls

- In Python, `Operation.wait()` with no arguments blocks forever -- this is usually what you want.
- `wait(timeout_ns=0)` returns `None` on timeout, which is distinct from a successful result. For operations that return `None` on success (like `remove_usd`), use the `Operation` object's state to distinguish.
- In C, `ovrtx_wait_op` waits for all operations up to and including the given `op_id`, not just that single operation.
- In C, check the `ovrtx_wait_op` return status before interpreting `wait_result` fields.
- Error strings from `ovrtx_get_last_op_error()` are valid only until the next `ovrtx_wait_op` call on the same thread.

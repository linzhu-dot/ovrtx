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

```python
op = renderer.add_usd_async("/path/to/scene.usda")

# Poll without blocking
result = op.wait(timeout_ns=0)
if result is None:
    print("Still loading...")
else:
    print(f"Loaded, handle: {result}")

# Or wait with a custom timeout (5 seconds)
result = op.wait(timeout_ns=5_000_000_000)
```

### Non-blocking step with two-stage timeout

```python
result = renderer.step_async(
    render_products={"/Render/Camera"},
    delta_time=1.0/60
)

# Wait with independent timeouts for step and fetch
products = result.wait(
    step_timeout_ns=5_000_000_000,   # 5s for raytracing
    fetch_timeout_ns=100_000_000     # 100ms for memory transfer
)

if products is None:
    if not result.step_complete:
        print("Step timed out")
    else:
        print("Fetch timed out")
```

### Infinite wait (default)

```python
# These are equivalent:
products = renderer.step(render_products={"/Render/Camera"}, delta_time=1.0/60)
# and:
result = renderer.step_async(render_products={"/Render/Camera"}, delta_time=1.0/60)
products = result.wait()
```

## C

### Poll with zero timeout

```c
ovrtx_op_wait_result_t wait_result;
ovrtx_result_t result = ovrtx_wait_op(
    renderer, op_id, ovrtx_timeout_t{0}, &wait_result);

if (result.status == OVRTX_API_TIMEOUT || wait_result.lowest_pending_op_id != 0) {
    // Still pending (non-blocking poll)
} else if (result.status == OVRTX_API_SUCCESS) {
    // All operations up to op_id are complete
} else {
    ovx_string_t error = ovrtx_get_last_error();
    fprintf(stderr, "wait_op failed: %.*s\n", (int)error.length, error.ptr);
}
```

### Poll loop with sleep

```c
ovrtx_op_wait_result_t wait_result;
do {
    ovrtx_result_t result =
        ovrtx_wait_op(renderer, op_id, ovrtx_timeout_t{0}, &wait_result);
    if (result.status == OVRTX_API_ERROR) {
        ovx_string_t error = ovrtx_get_last_error();
        fprintf(stderr, "wait_op failed: %.*s\n", (int)error.length, error.ptr);
        break;
    }
    if (wait_result.lowest_pending_op_id != 0) {
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
} while (wait_result.lowest_pending_op_id != 0);
```

### Block indefinitely

```c
ovrtx_op_wait_result_t wait_result;
ovrtx_result_t result = ovrtx_wait_op(
    renderer, op_id, ovrtx_timeout_infinite, &wait_result);
```

### Check for errors after wait

```c
if (wait_result.num_error_ops > 0) {
    for (size_t i = 0; i < wait_result.num_error_ops; ++i) {
        ovrtx_op_id_t failed_id = wait_result.error_op_ids[i];
        ovx_string_t error = ovrtx_get_last_op_error(failed_id);
        fprintf(stderr, "Op %llu failed: %.*s\n",
                (unsigned long long)failed_id,
                (int)error.length, error.ptr);
    }
}
```

## Key Types / Functions

| Python | C |
|--------|---|
| `Operation.wait(timeout_ns=None)` | `ovrtx_wait_op(renderer, op_id, timeout, &wait_result)` |
| `Operation.wait(timeout_ns=0)` | `ovrtx_wait_op` with `timeout.time_out_ns = 0` |
| `RendererResult.wait(step_timeout_ns, fetch_timeout_ns)` | manual two-stage: `ovrtx_wait_op` then `ovrtx_fetch_results` |
| `RuntimeError` on failure | `wait_result.num_error_ops > 0` |

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

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
name: application-flow
description: High-level overview of a typical ovrtx application lifecycle. Use when user asks how to structure an ovrtx program, what the main steps are, or how the pieces fit together.
---

# Application Flow

## Overview

Every ovrtx application follows the same core lifecycle, whether in Python or C. This skill gives the high-level sequence and points to the detailed skill for each step.

```
1. Create renderer         → renderer-creation
2. Load USD scene(s)       → loading-usd
3. [Optional] Clone prims  → cloning-prims
4. Render loop:
   a. Write attributes     → writing-transforms, writing-attributes
   b. Step renderer        → stepping-and-rendering
   c. Read output          → reading-render-output
5. Cleanup
```

## Python

> **Source:** `examples/python/minimal/main.py` snippet `create-renderer`
>
> Followed by: `examples/python/minimal/main.py` snippet `add-usd`
>
> Followed by: `examples/python/minimal/main.py` snippet `step`
>
> Followed by: `examples/python/minimal/main.py` snippet `read-render-output`
>
> For the full lifecycle with attribute writes, bindings, and cloning, compose the relevant skill snippets.

## C

> **Source:** `examples/c/minimal/main.cpp` snippet `create-renderer`
>
> Followed by: `examples/c/minimal/main.cpp` snippet `load-usd-and-wait`
>
> Followed by: `examples/c/minimal/main.cpp` snippet `step-renderer`
>
> Followed by: `examples/c/minimal/main.cpp` snippet `fetch-results`
>
> Followed by: `examples/c/minimal/main.cpp` snippet `map-rendered-output-cpu`
>
> Followed by: `examples/c/minimal/main.cpp` snippet `unmap-and-cleanup`

## Key Differences: Python vs C

| Aspect | Python | C |
|--------|--------|---|
| Renderer lifetime | GC or explicit `del` | `ovrtx_destroy_renderer()` |
| USD loading | `add_usd()` blocks | `ovrtx_add_usd()` is async; must poll/wait |
| Step | `step()` returns outputs directly | `ovrtx_step()` + `ovrtx_wait_op()` + `ovrtx_fetch_results()` |
| Output access | Context manager `with var.map()` | `ovrtx_map_rendered_output()` + `ovrtx_unmap_rendered_output()` |
| Result cleanup | Automatic (GC / context manager) | Must call `ovrtx_destroy_results()` |
| Error handling | Python exceptions (`RuntimeError`) | Check `ovrtx_result_t.status` + `ovrtx_get_last_error()` |

## Common Pitfalls

- In C, **every async operation** (`ovrtx_add_usd`, `ovrtx_clone_usd`, `ovrtx_step`) returns an `ovrtx_enqueue_result_t`. You must wait on the `op_index` before using the results.
- In C, **always destroy results** with `ovrtx_destroy_results()` after processing. ovrtx will log warnings if results are leaked.
- In Python, the synchronous API (`add_usd`, `step`, `clone_usd`) handles waiting internally. Use the `_async` variants for explicit control.
- For best performance in animation loops, use **attribute bindings** (see `attribute-bindings` skill) or **mapping** (see `mapping-attributes` skill) instead of per-frame `write_attribute` calls.
- See `error-handling` skill for robust error checking patterns in both languages.

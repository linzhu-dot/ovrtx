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

```python
import ovrtx
import numpy as np
from PIL import Image

# 1. Create renderer (see renderer-creation skill)
renderer = ovrtx.Renderer()

# 2. Load USD (see loading-usd skill)
renderer.add_usd("scene.usda")

# 3. (Optional) Clone prims for instancing (see cloning-prims skill)
renderer.clone_usd("/World/Template", [f"/World/Copy_{i}" for i in range(10)])

# 4. Render loop
for frame_idx in range(100):
    # 4a. Write attributes (see writing-transforms, writing-attributes skills)
    #     For repeated writes, use attribute bindings (see attribute-bindings skill)
    #     For zero-copy GPU writes, use mapping (see mapping-attributes skill)

    # 4b. Step (see stepping-and-rendering skill)
    products = renderer.step(
        render_products={"/Render/Camera"},
        delta_time=1.0 / 60,
    )

    # 4c. Read output (see reading-render-output skill)
    for product_name, product in products.items():
        for frame in product.frames:
            with frame.render_vars["LdrColor"].map(device="cpu") as var:
                pixels = var.tensor.numpy()
                Image.fromarray(pixels).save(f"frame_{frame_idx:04d}.png")

# 5. Cleanup -- Python garbage collection handles renderer teardown,
#    but explicit cleanup is recommended for bindings, etc.
```

## C

```c
#include <ovrtx/ovrtx.h>
#include <ovrtx/ovrtx_config.h>
#include <ovrtx/ovrtx_types.h>

#include <cstring>

// 1. Create renderer (see renderer-creation skill)
ovrtx_renderer_t* renderer = nullptr;
ovrtx_config_t config = {};
ovrtx_result_t result = ovrtx_create_renderer(&config, &renderer);

// 2. Load USD (see loading-usd skill)
ovrtx_usd_handle_t usd_handle = 0;
ovrtx_usd_input_t usd_input = {};
usd_input.usd_file_path = {"scene.usda", strlen("scene.usda")};

ovrtx_enqueue_result_t enqueue_result =
    ovrtx_add_usd(renderer, usd_input, {"", 0}, &usd_handle);

// Wait for load (async in C -- see loading-usd skill)
ovrtx_op_wait_result_t wait_result;
while (ovrtx_wait_op(renderer, enqueue_result.op_index, ovrtx_timeout_t{0},
                     &wait_result).status == OVRTX_API_TIMEOUT) {
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
}

// 3. (Optional) Clone prims (see cloning-prims skill)
//    Must wait on the clone op before using the new prims.

// 4. Render loop
ovx_string_t rp_path = {"/Render/Camera", strlen("/Render/Camera")};
ovrtx_render_product_set_t render_products = {};
render_products.render_products = &rp_path;
render_products.num_render_products = 1;

for (int frame = 0; frame < 100; ++frame) {
    // 4a. Write attributes (see writing-transforms, writing-attributes skills)
    //     For repeated writes, use attribute bindings (see attribute-bindings skill)
    //     For zero-copy writes, use mapping (see mapping-attributes skill)

    // 4b. Step (see stepping-and-rendering skill)
    ovrtx_step_result_handle_t step_handle = 0;
    enqueue_result =
        ovrtx_step(renderer, render_products, 1.0 / 60.0, &step_handle);
    ovrtx_wait_op(renderer, enqueue_result.op_index,
                  ovrtx_timeout_infinite, &wait_result);

    // 4c. Fetch + read output (see reading-render-output skill)
    ovrtx_render_product_set_outputs_t outputs = {};
    ovrtx_fetch_results(renderer, step_handle,
                        ovrtx_timeout_infinite, &outputs);

    // ... map output, read pixels, unmap ...

    // Free step results each iteration
    ovrtx_destroy_results(renderer, step_handle);
}

// 5. Cleanup
ovrtx_destroy_renderer(renderer);
```

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

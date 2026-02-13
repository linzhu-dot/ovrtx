// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: LicenseRef-NvidiaProprietary
//
// NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
// property and proprietary rights in and to this material, related
// documentation and any modifications thereto. Any use, reproduction,
// disclosure or distribution of this material and related documentation
// without an express license agreement from NVIDIA CORPORATION or
// its affiliates is strictly prohibited.

#include <ovrtx/ovrtx_config.h>
#include <ovrtx/ovrtx_types.h>
#include <ovrtx/ovrtx.h>

#include <cstring>
#include <thread>
#include <stdexcept>

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

#define THROW_ON_ERROR(RESULT, OPERATION)                                      \
    do {                                                                       \
        if (RESULT.status == OVRTX_API_ERROR) {                              \
            ovx_string_t error = ovrtx_get_last_error();                       \
            char error_msg[512];                                               \
            if (error.ptr && error.length > 0) {                               \
                snprintf(error_msg,                                            \
                         sizeof(error_msg),                                    \
                         "ovrtx %s failed: %.*s",                              \
                         OPERATION,                                            \
                         static_cast<int>(error.length),                       \
                         error.ptr);                                           \
            } else {                                                           \
                snprintf(error_msg,                                            \
                         sizeof(error_msg),                                    \
                         "ovrtx %s failed",                                    \
                         OPERATION);                                           \
            }                                                                  \
            throw std::runtime_error(error_msg);                               \
        }                                                                      \
    } while (0)

// Find the handle of the given output in the given set of outputs
static ovrtx_rendered_output_handle_t
find_output(ovrtx_render_product_set_outputs_t const& outputs,
            char const* output_to_find) {
    ovrtx_rendered_output_handle_t output_handle = -1;
    for (size_t i = 0; i < outputs.output_count; ++i) {
        ovrtx_render_product_output_t const& product_output =
            outputs.outputs[i];
        for (size_t f = 0; f < product_output.output_frame_count; ++f) {
            ovrtx_render_product_frame_output_t const& frame =
                product_output.output_frames[f];
            for (size_t v = 0; v < frame.render_var_count; ++v) {
                ovrtx_render_product_render_var_output_t const& var =
                    frame.output_render_vars[v];
                if (var.render_var_name.ptr &&
                    strncmp(var.render_var_name.ptr,
                            output_to_find,
                            var.render_var_name.length) == 0) {
                    output_handle = var.output_handle;
                    break;
                }
            }
        }
    }
    if (output_handle == -1) {
        throw std::runtime_error("LdrColor output not found");
    }

    return output_handle;
}

int main() {
    ovrtx_renderer_t* renderer = nullptr;
    ovrtx_result_t result;

    // Create the renderer, optionally providing configuration settings.
    // In this case we need no configuration.
    ovrtx_config_t config {};
    result = ovrtx_create_renderer(&config, &renderer);
    THROW_ON_ERROR(result, "create_renderer");

    // Load a USD layer into the renderer.
    //
    // As well as just passing a URI to an existing layer, we can pass a USDA
    // string in order to compose a Stage at runtime. This can be very useful
    // for dynamically creating the RenderProducts etc. that define the render
    // output rather than editing the original layer to add them.
    //
    // A real application might want to load the USD layer and traverse it to
    // find either existing RenderProducts, and/or Cameras and allow the user to
    // select which one to render, and which RenderVars to output.
    //
    // Here, we know there's a suitable camera at /World/Camera, so we'll just
    // create a RenderProduct referencing that and add the LdrColor RenderVar to
    // get some output.
    ovrtx_usd_handle_t usd_handle{};
    ovrtx_usd_input_t usd_input{};
    char const* usd_url = "https://omniverse-content-production.s3.us-west-2.amazonaws.com/Samples/Robot-OVRTX/robot-ovrtx.usda";
    usd_input.usd_file_path = {usd_url, strlen(usd_url)};

    ovrtx_enqueue_result_t enqueue_result =
        ovrtx_add_usd(renderer, usd_input, {"", 0}, &usd_handle);

    // This operation is asynchronous as loading the USD may take a long time.
    // We'll just poll every 100ms till it's done.
    ovrtx_op_wait_result_t wait_result;
    result = ovrtx_wait_op(
        renderer, enqueue_result.op_index, ovrtx_timeout_t{0}, &wait_result);
    THROW_ON_ERROR(result, "wait_op");
    while (ovrtx_wait_op(renderer, enqueue_result.op_index, ovrtx_timeout_t{0}, 
                        &wait_result).status == OVRTX_API_TIMEOUT) {
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }


    // We render a frame by stepping the renderer.
    //
    // Any sensors whose exposures end during this step will generate a frame
    // that will be available in the step result. Since the camera in the loaded
    // USD layer is instantaneous (does not specify motion blur or rolling
    // shutter), it will generate a frame every time the render is stepped.
    //
    // To step the renderer we need to tell ovrtx which RenderProducts we're
    // interested in, which in this case is the RenderProduct we defined in the
    // loading layer.
    ovrtx_render_product_set_t render_products = {};
    ovx_string_t render_product_str = {"/Render/Camera", strlen("/Render/Camera")};
    render_products.render_products = &render_product_str;
    render_products.num_render_products = 1;

    ovrtx_step_result_handle_t step_result_handle = 0;
    enqueue_result =
        ovrtx_step(renderer, render_products, 1.0 / 60.0, &step_result_handle);
    THROW_ON_ERROR(enqueue_result, "step");

    // Wait for the render to complete. Here we'll just block until it's done.
    result = ovrtx_wait_op(renderer,
                           enqueue_result.op_index,
                           ovrtx_timeout_infinite,
                           &wait_result);
    THROW_ON_ERROR(result, "wait_op");

    ovrtx_render_product_set_outputs_t outputs = {};
    result = ovrtx_fetch_results(
        renderer, step_result_handle, ovrtx_timeout_infinite, &outputs);
    THROW_ON_ERROR(result, "fetch_results");

    // Find LdrColor in outputs
    ovrtx_rendered_output_handle_t ldrcolor_output_handle =
        find_output(outputs, "LdrColor");

    // Map rendered output so that it can be accessed on the CPU
    ovrtx_map_output_description_t map_desc = {};
    map_desc.device_type = OVRTX_MAP_DEVICE_TYPE_CPU;
    ovrtx_rendered_output_t rendered_output = {};
    result = ovrtx_map_rendered_output(renderer,
                                       ldrcolor_output_handle,
                                       &map_desc,
                                       ovrtx_timeout_infinite,
                                       &rendered_output);
    THROW_ON_ERROR(result, "map_rendered_output");

    // The output is returned as a DLTensor
    DLTensor const& tensor = rendered_output.buffer.dl;
    int width = static_cast<int>(tensor.shape[1]);
    int height = static_cast<int>(tensor.shape[0]);
    stbi_write_png("out.png",
                   width,
                   height,
                   /* components = */ 4,
                   tensor.data,
                   /* row stride in bytes = */ 4 * width);

    // Unmap output
    ovrtx_cuda_sync_t no_sync = {};
    result = ovrtx_unmap_rendered_output(
        renderer, rendered_output.map_handle, no_sync);
    THROW_ON_ERROR(result, "unmap_rendered_output");

    // Clean up resources (ovrtx will warn if results are leaked)
    result = ovrtx_destroy_results(renderer, step_result_handle);
    result = ovrtx_destroy_renderer(renderer);
    THROW_ON_ERROR(result, "destroy_renderer");

    return 0;
}

/* Copyright (c) 2025-2026, NVIDIA CORPORATION. All rights reserved.
 *
 * NVIDIA CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA CORPORATION is strictly prohibited.
 */

#ifndef OVRTX_H
#define OVRTX_H

#include "ovrtx_types.h"

#ifdef __cplusplus
extern "C"
{
#endif

    /** @file
     * General notes
     * Return values:
     *   All operations return status of **OVRTX_API_SUCCESS** if the operation was enqueued successfully,
     *   **OVRTX_API_ERROR** if the operation failed to be enqueued or already failed to execute before
     *   the enqueue call returned.
     *   If an error is returned the error string is valid until the next API call from the current thread.
     *   Note that the error data lives in thread local storage so any operation that might trigger a fiber
     *   switch will also invalidate the error string.
     * Stream ordered asynchronous execution:
     *   All operations that are marked as asynchronous execute in a stream-ordered fashion.
     *   They might return handles that can be used in subsequent operations,
     *   but the actual objects or effects represented by those handles will not necessarily be produced
     *   when the enqueue function calls return.
     *   Stream synchronization events can be used to ensure stream execution for an enqueued function call and all prior calls
     *   have happened.
     *   Specifically a synchronization event ensures that input resources provided to previously enqueued operations
     *   will no longer be accessed after the synchronization event is signaled.
     *   Synchronization events can be device specific to allow ensuring that access to input resources on a certain device
     *   are completed before the synchronization event is signaled.
     *
     *   When a stream ordered operation is executed, the result of the operation will look as if all prior operations
     *   have been executed and completed serially. The underlying execution of operations might overlap, when the system
     *   determines that this will not affect the result of the operation.
     */

    /** @defgroup ovrtx_version Version query
     *  @{
     */

    /**
     * Get the ovrtx API version that the library was compiled with.
     *
     * This function can be called at any time, including before ovrtx_initialize().
     * It can be used to verify that the loaded library matches the header version at runtime.
     *
     * The compile-time version is also available via OVRTX_VERSION_MAJOR, OVRTX_VERSION_MINOR,
     * and OVRTX_VERSION_PATCH macros.
     *
     * @param[out] out_major Major version number
     * @param[out] out_minor Minor version number
     * @param[out] out_patch Patch version number
     */
    void ovrtx_get_version(uint32_t* out_major, uint32_t* out_minor, uint32_t* out_patch);

    /** @} */ // end of ovrtx_version

    /** @defgroup ovrtx_creation Creation and destruction operations
     *  @{
     */

    /**
     * Initialize the ovrtx loader or increase its ref count. 
     * 
     * It is allowed to call this function multiple times and for each successful call a corresponding call to ovrtx_shutdown() 
     * is required.
     * 
     * Note that explicit initialization is not required: creating a render instance with ovrtx_create_renderer() will also
     * initialize the system if needed. Calling ovrtx_initialize() and ovrtx_shutdown() can be used
     * to prevent system shutdown and initialization if the renderer is recreated multiple times.
     * @param config Configuration for the ovrtx system
     * @return 
     * - **OVRTX_API_SUCCESS** if the system was initialized or ref-count was increased successfully,
     * - **OVRTX_API_ERROR** if initialization failed.
     */
    ovrtx_result_t ovrtx_initialize(const ovrtx_config_t* config);

    /**
     * Shuts down the ovrtx system or decreases it's ref count. One call per call to ovrtx_initialize() is required.
     * Note that any render instances that have not been destroyed will keep the system alive until they are destroyed.
     * @return 
     * - **OVRTX_API_SUCCESS** if the system was released successfully (though might still be loaded),
     * - **OVRTX_API_ERROR** if the system shutdown failed or was not initialized.
     */
    ovrtx_result_t ovrtx_shutdown();

    /**
     * Create a new renderer instance. System initialization is done automatically if ovrtx_initialize() has not
     * been called yet, but in that case the config must contain both initialization and renderer settings.
     * The system will be kept running until a corresponding call to ovrtx_destroy_renderer().
     * @param config Configuration for the renderer (see @ref ovrtx_config.h "ovrtx_config.h"):
     * - "binary_package_root_path" (STRING): path to the OVRTX binary package root directory.
     * - "sync_mode" (BOOL): if true, stream operations execute synchronously (enqueue blocks).
     * - "enable_profiling" (BOOL): if true, enables internal profiling.
     * - "read_gpu_transforms" (BOOL): if true, uses GPU world transform propagation during rendering.
     * - "log_file_path" (STRING): log file path for carb logging. The crash dump directory
     *   is automatically set to the same directory as the log file.
     * - "log_level" (STRING): log level string for carb logging (e.g., "verbose", "info", "warn", "error").
     * @param out_renderer [out] Renderer instance
     * @return 
     * - **OVRTX_API_SUCCESS** if the renderer was created successfully,
     * - **OVRTX_API_ERROR** if the renderer creation failed.
     */
    ovrtx_result_t ovrtx_create_renderer(const ovrtx_config_t* config, ovrtx_renderer_t** out_renderer);

    /**
     * Destroy a renderer instance.
     * @param renderer Renderer instance to destroy
     * @return 
     * - **OVRTX_API_SUCCESS** if the renderer was destroyed successfully,
     * - **OVRTX_API_ERROR** if the renderer destruction failed.
     */
    ovrtx_result_t ovrtx_destroy_renderer(ovrtx_renderer_t* renderer);

    /** @} */ // end of ovrtx_creation

    /** @defgroup ovrtx_stage_building Stage building operations
     *  @{
     */

    /**
     * Enqueue an asynchronous operation to add a USD file to the runtime stage representation.
     *
     * Exactly one of the fields in the @ref ovrtx_usd_input_t must be set. 
     *
     * `ovrtx_add_usd()` will add the given USD input as a reference at the given path prefix. An empty path prefix will
     * instead result in the USD file being added as a sublayer.
     *
     * Note that errors occuring during loading (including a given USD file not being found) will be reported through the
     * @ref ovrtx_op_wait_result_t::error_op_ids list.
     *
     * @param instance Renderer instance
     * @param usd_input Input to add a USD file to the runtime stage representation
     * @param path_prefix Prefix path that will be added to all paths in the USD file.
     *                    This is useful when adding multiple USD files, so that the
     *                    prims in each USD file have unique, known paths.
     *                    If a combined prefix path and prim path inside the loaded usd file already exists an
     *                    error will be returned and the usd file will not be added.
     * @param out_add_usd_handle [out] Handle to the added usd file to be used with @ref ovrtx_remove_usd().
     * @return 
     * - **OVRTX_API_SUCCESS** if the operation was enqueued successfully.
     * - **OVRTX_API_ERROR** if the operation was not enqueued successfully.
     */
    ovrtx_enqueue_result_t ovrtx_add_usd(ovrtx_renderer_t* instance,
                                            ovrtx_usd_input_t usd_input,
                                            ovx_string_t path_prefix,
                                            ovrtx_usd_handle_t* out_add_usd_handle);

    /**
     * Enqueue an asynchronous operation to remove a prior added usd file from the runtime stage representation.
     * All prims added to the stage during ovrtx_add_usd() will be removed from the stage.
     * @param instance Renderer instance
     * @param add_usd_handle Handle obtained from add_usd to identify the usd file to remove
     * @return 
     * - **OVRTX_API_SUCCESS** if the usd file was removed successfully,
     * - **OVRTX_API_ERROR** if the usd file removal failed.
     */
    ovrtx_enqueue_result_t ovrtx_remove_usd(ovrtx_renderer_t* instance,
                                            ovrtx_usd_handle_t add_usd_handle);

    /**
     * Enqueue an asynchronous operation to clone the subtree under the source path to
     * one or more target paths in the runtime stage representation.
     * The source path must exist in the stage.
     * The target paths must not already exist in the stage.
     * @param instance Renderer instance
     * @param source_path_in_usd Path to the source path to clone
     * @param target_paths Array of target paths to clone to
     * @param num_target_paths Number of target paths to clone to
     * @return 
     * - **OVRTX_API_SUCCESS** if the path was cloned successfully,
     * - **OVRTX_API_ERROR** if the path cloning failed.
     */
    ovrtx_enqueue_result_t ovrtx_clone_usd(ovrtx_renderer_t* instance,
                                           ovx_string_t source_path_in_usd,
                                           const ovx_string_t* target_paths,
                                           size_t num_target_paths);

    /**
     * Enqueue an asynchronous operation to reset the runtime stage representation to an empty stage.
     * @param instance Renderer instance
     * @return 
     * - **OVRTX_API_SUCCESS** if the stage was reset successfully,
     * - **OVRTX_API_ERROR** if the stage reset failed.
     */
    ovrtx_enqueue_result_t ovrtx_reset_stage(ovrtx_renderer_t* instance);

    /**
     * Enqueue an asynchronous operation to update the runtime stage representation from a specific USD time.
     * This operation will update all time-sampled attributes in the runtime stage representation to the provided USD time.
     * @param instance Renderer instance
     * @param usd_time USD time to update the stage to
     * @return 
     * - **OVRTX_API_SUCCESS** if the stage was updated from the USD time successfully,
     * - **OVRTX_API_ERROR** if the stage update from USD time failed.
     */
    ovrtx_enqueue_result_t ovrtx_update_stage_from_usd_time(ovrtx_renderer_t* instance, double usd_time);

    /** @} */ // end of ovrtx_stage_building

    /** @defgroup ovrtx_attribute_write Stage attribute write operations
     *  @{
     */

    /**
     * Enqueue an asynchronous write operation from source data into the system's stage representation.
     * This write operation is not fully executed when the call returns and inputs with asynchronous access
     * must remain valid until the stream execution of this operation has completed.
     * @param instance Renderer instance
     * @param binding_handle_or_desc Handle or description of the binding to write to.
     *                    This binding defines the layout of the input data, both in terms
     *                    of attribute data encoding as well as the layout of prims. The binding
     *                    can be generated in place or a persistent handle can be provided to manually
     *                    manage the lifetime.
     * @param data_array Source data to write. Based on the input_access this source is used during the execution of the write operation,
     *                    not during the enqueue. So it is important that the data source remains valid
     *                    until the write operation has completed.
     *                    This can be determined through stream synchronization events using
     *                    ovrtx_signal_event() and ovrtx_wait_all_events().
     *                    When using asynchronous access of GPU input data, the cuda synchronization event
     *                    must be signaled when the input data is ready to be accessed.
     * @param data_access Determines the time of access to the input data.
     *                    With asynchronous access, the lifetime must be managed by the user,
     *                    while synchronous access incurs a synchronous copy during this call but
     *                    prevents any access after this call returns.
     * @return 
     * - **OVRTX_API_SUCCESS** if the attribute was written successfully,
     * - **OVRTX_API_ERROR** if the attribute write failed.
     */
    ovrtx_enqueue_result_t ovrtx_write_attribute(ovrtx_renderer_t* instance,
                                                const ovrtx_binding_desc_or_handle_t* binding_handle_or_desc,
                                                const ovrtx_input_buffer_t* data_array,
                                                ovrtx_data_access_t data_access);

    /**
     * Immediately provedes internal memory according to the binding description
     * to be written to by the user and later applied to the stage representation
     * via ovrtx_unmap_attribute()
     * @param instance Renderer instance
     * @param binding_handle_or_desc Handle or description of the binding to use to determine
     *                    the layout of the data to write. The binding
     *                    can be generated in place or a persistent handle can be provided to manually
     *                    manage the lifetime.
     * @param mapping_desc Description of the mapping to use
     * @param out_attribute_mapping [out] Handle to the attribute mapping
     * @return 
     * - **OVRTX_API_SUCCESS** if the attribute was mapped successfully,
     * - **OVRTX_API_ERROR** if the attribute mapping failed.
     */
    ovrtx_result_t ovrtx_map_attribute(ovrtx_renderer_t* instance,
                                        const ovrtx_binding_desc_or_handle_t* binding_handle_or_desc,
                                        ovrtx_mapping_desc_t mapping_desc,
                                        ovrtx_attribute_mapping_t* out_attribute_mapping);

    /**
     * Enqueue an asynchronous operation to take the data written by the user and do whatever necessary to
     * apply it to the system's stage representation.
     * Note that while the map operation is not asynchronous, the unmap operation is and it determines
     * the logical order of applying the written data to the stage.
     * Multiple mappings can be outstanding on the same stage data with the effects on the stage representation
     * depending on the order of the unmap operations.
     * The data written by the user must be ready when the unmap operation is called for CPU data and when
     * the cuda synchronization event is signaled for GPU data.
     * @param instance Renderer instance
     * @param map_handle Handle to the attribute mapping to unmap
     * @param cuda_sync optional cuda synchronization to wait for before the mapped memory is accessed during the
     *                   application of the written data to the stage representation.
     * @return 
     * - **OVRTX_API_SUCCESS** if the attribute was unmapped successfully,
     * - **OVRTX_API_ERROR** if the attribute unmap failed.
     */
    ovrtx_enqueue_result_t ovrtx_unmap_attribute(ovrtx_renderer_t* instance,
                                                ovrtx_map_handle_t map_handle,
                                                ovrtx_cuda_sync_t cuda_sync);

    /**
     * Enqueue an asynchronous operation to create a persistent attribute binding that binds
     * a list of prims to a buffer layout.
     * This operation is an optimization to manage the lifetime of internal resources
     * used to perform write or map operations using this binding.
     * @param instance Renderer instance
     * @param description Description of the binding to create
     * @param out_attribute_binding_handle [out] Handle to the attribute binding
     * @return 
     * - **OVRTX_API_SUCCESS** if the attribute binding was created successfully,
     * - **OVRTX_API_ERROR** if the attribute binding creation failed.
     */
    ovrtx_enqueue_result_t ovrtx_create_attribute_binding(ovrtx_renderer_t* instance,
                                                            const ovrtx_binding_desc_t* description,
                                                            ovrtx_attribute_binding_handle_t* out_attribute_binding_handle);

    /**
     * Enqueue an asynchronous operation to destroy a persistent attribute binding.
     * @param instance Renderer instance
     * @param binding_handle Handle to the attribute binding to destroy
     * @return 
     * - **OVRTX_API_SUCCESS** if the attribute binding was destroyed successfully,
     * - **OVRTX_API_ERROR** if the attribute binding destruction failed.
     */
    ovrtx_enqueue_result_t ovrtx_destroy_attribute_binding(ovrtx_renderer_t* instance,
                                                        ovrtx_attribute_binding_handle_t binding_handle);

    /** @} */ // end of ovrtx_attribute_write

    /** @defgroup ovrtx_sensor_simulation Sensor simulation operations
     *  @{
     */

    /**
     * Enqueue an asynchronous operation that will perform a sensor simulation step for all render products
     * in the provided render product set. The simulation step will be performed for the time span
     * [last_step_time, last_step_time + delta_time], where last_step_time is determined
     * by the history of previous calls to ovrtx_step() or ovrtx_reset().
     * When performing the sensor simulation, the result of prior stream ordered operations
     * affecting the stage since the last call of ovrtx_step() will be considered the state of the stage
     * at time (last_step_time + delta_time).
     * After the simulation step was executed, last_step_time will be updated to (last_step_time + delta_time)
     * for the next call to ovrtx_step().
     * @param instance Renderer instance
     * @param render_products Render products to simulate during this simulation step.
     *                        Accumulated sensor rendering history for all render products not in the
     *                        provided set will be discarded.
     * @param delta_time Time step to simulate
     * @param out_step_result_handle [out] Handle to the step result
     * @return 
     * - **OVRTX_API_SUCCESS** if the step was enqueued successfully,
     * - **OVRTX_API_ERROR** if the step enqueue failed.
     */
    ovrtx_enqueue_result_t ovrtx_step(ovrtx_renderer_t* instance,
                                    ovrtx_render_product_set_t render_products,
                                    double delta_time,
                                    ovrtx_step_result_handle_t* out_step_result_handle);

    /**
     * Enqueue an asynchronous operation to reset the accumulated sensor rendering history for all
     * render products and start future sensor simulation steps at the provided time.
     * After the reset was executed, last_step_time will be updated to the provided time for the next call to ovrtx_step().
     * @param instance Renderer instance
     * @param time Time to reset the simulation to
     * @return 
     * - **OVRTX_API_SUCCESS** if the reset was enqueued successfully,
     * - **OVRTX_API_ERROR** if the reset enqueue failed.
     */
    ovrtx_enqueue_result_t ovrtx_reset(ovrtx_renderer_t* instance,
                                    double time);

    /**
     * Query the results of a prior enqueued step operation.
     * This operation is synchronous and will block until the results are available or the timeout
     * has passed. By passing 0 as the timeout this operation becomes a non-blocking poll operation
     * that returns immediately if the results are not yet available.
     * The complete production of render outputs is not determined by the completion of the
     * asynchronous ovrtx_step() operation within the stream, so it is not possible to ensure this
     * operations returns the results immediately by waiting for a stream synchronization event
     * signaled after the ovrtx_step() operation.
     * The result of this operation contains information about what results each render product has produced.
     * Each render product can have produced 0-n frames of output for m render vars.
     * This operation doesn't return the actual output data, but rather a handle to the output data which can then
     * be mapped to retrieve the actual output data.
     * All strings and pointers inside the result are valid until the result is destroyed by ovrtx_destroy_results().
     * @param instance Renderer instance
     * @param result_handle Handle to the step result to query
     * @param timeout Timeout for the operation.
     *                Passing 0 will make the operation non-blocking and return immediately with the current status of the operation.
     * @param out_render_product_set_outputs [out] Render product set outputs
     * @return 
     * - **OVRTX_API_SUCCESS** if the render product set outputs were retrieved successfully,
     * - **OVRTX_API_ERROR** if the operation failed,
     * - **OVRTX_API_TIMEOUT** if the result could not be obtained within the timeout.
     */
    ovrtx_result_t ovrtx_fetch_results(ovrtx_renderer_t* instance,
                                        ovrtx_step_result_handle_t result_handle,
                                        ovrtx_timeout_t timeout,
                                        ovrtx_render_product_set_outputs_t* out_render_product_set_outputs);

    /**
     * Maps the rendered output into user accessible memory when the result are available.
     * This operation is synchronous and will block until the output is mapped or the timeout has passed.
     * By passing 0 as the timeout this operation becomes a non-blocking poll operation
     * that returns immediately if the output is not yet mapped.
     * The complete production of render outputs is not determined by the completion of the
     * asynchronous ovrtx_step() operation within the stream, so it is not possible to ensure this
     * operations returns the results immediately by waiting for a stream synchronization event
     * signaled after the ovrtx_step() operation.
     * The result of this operation contains the actual rendered output memory, but it can contain cuda events that must be
     * synchronized to before actually accessing the output memory.
     * Calling map on a output_handle that is part of a step result after calling destroy_results on
     * the step result will return an error.
     * @param instance Renderer instance
     * @param output_handle Handle to the output to map
     * @param map_output_desc Description of the output to map
     * @param timeout Timeout for the operation.
     *                Passing 0 will make the operation non-blocking and return immediately with the current status of the operation.
     * @param out_rendered_output [out] Rendered output
     * @return 
     * - **OVRTX_API_SUCCESS** if the output was mapped successfully.
     * - **OVRTX_API_ERROR** if the output mapping failed.
     * - **OVRTX_API_TIMEOUT** if the result could not be obtained within the timeout.
     */
    ovrtx_result_t ovrtx_map_rendered_output(ovrtx_renderer_t* instance,
                                            ovrtx_rendered_output_handle_t output_handle,
                                            const ovrtx_map_output_description_t* map_output_desc,
                                            ovrtx_timeout_t timeout,
                                            ovrtx_rendered_output_t* out_rendered_output);

    /**
     * Unmaps the rendered output and frees resources associated with the prior map_rendered_output.
     * When this is called all access to the buffer provided by map_rendered_output must be done.
     * This call determines the lifetime of the resources made accessible through the prior map_rendered_output call and
     * this lifetime is independent of ovrtx_destroy_results().
     * It is safe to call unmap on a map_handle that was produced from a step result that has been destroyed by destroy_results.
     * @param instance Renderer instance.
     * @param map_handle Handle to the map to unmap.
     * @param before_destroy_cuda_sync CUDA synchronization to wait for before the mapped memory is destroyed.
     * @return 
     * - **OVRTX_API_SUCCESS** if the output was unmapped successfully,
     * - **OVRTX_API_ERROR** if the output unmapping failed.
     */
    ovrtx_result_t ovrtx_unmap_rendered_output(ovrtx_renderer_t* instance,
                                                ovrtx_rendered_output_map_handle_t map_handle,
                                                ovrtx_cuda_sync_t before_destroy_cuda_sync);

    /**
     * Releases all resources associated with the result of a sensor simulation step, with the exception of resources provided by calls to
     * map_rendered_output. Those are only released through unmap_rendered_output.
     * @param instance Renderer instance
     * @param result_handle Handle to the step result to destroy
     * @return 
     * - **OVRTX_API_SUCCESS** if the step result was destroyed successfully,
     * - **OVRTX_API_ERROR** if the step result destruction failed.
     */
    ovrtx_result_t ovrtx_destroy_results(ovrtx_renderer_t* instance,
                                          ovrtx_step_result_handle_t result_handle);

    /** @} */ // end of ovrtx_sensor_simulation

    /** @defgroup ovrtx_stream Stream operations
     *  @{
     */

    /**
     * Wait for completion of all operations up to and including the specified operation id.
     * This operation is synchronous and will block until the operations are completed or the timeout has passed.
     * Passing 0 as the timeout makes the operation a non-blocking poll.
     * The out structure returns any errors observed since the last wait call and, on timeout, the list of still-active op ids.
     * All error strings returned by this operation must be released by calling ovrtx_release_errors.
     * @param instance Renderer instance
     * @param op_id Operation id to wait for
     * @param time_out Timeout for the operation
     * @param out_wait_result [out] Wait result information (errors and active operation ids)
     * @return 
     * - **OVRTX_API_SUCCESS** if the operations were waited for successfully,
     * - **OVRTX_API_ERROR** if the wait failed (e.g., invalid op id),
     * - **OVRTX_API_TIMEOUT** if not all operations completed within the timeout.
     */
    ovrtx_result_t ovrtx_wait_op(ovrtx_renderer_t* instance,
            ovrtx_op_id_t op_id,
            ovrtx_timeout_t time_out,
            ovrtx_op_wait_result_t* out_wait_result);

    /** @} */ // end of ovrtx_stream

    /** @defgroup ovrtx_extension Extension querying
     *  @{
     */

    /**
     * Query for an internal extension interface by name.
     * @param name The name of the extension
     * @param vtable [out] Vtable with function pointers for the extension
     * @return 
     * - **OVRTX_API_SUCCESS** if the extension was queried successfully,
     * - **OVRTX_API_ERROR** if the extension is unavailable or if the system is not initialized yet.
     */
   ovrtx_result_t ovrtx_query_extension(const char* name,
        const void** vtable);

    /** @} */ // end of ovrtx_extension

    /** @defgroup ovrtx_error Error handling
     *  @{
     */

    /**
     * Returns the error string for the latest API call on the calling thread. The string is 
     * valid until the next API call (or any call that might perform a fiber switch) on the same thread.
     */
    ovx_string_t ovrtx_get_last_error();

    /**
     * Returns the error string for the provided operation id from the last call to ovrtx_wait_op  
     * on the calling thread. The string is valid until the next call to ovrtx_wait_op (or any call
     * that might perform a fiber switch) on the same thread.
     * @param op_id Operation id to get the error string for
     */
    ovx_string_t ovrtx_get_last_op_error(ovrtx_op_id_t op_id);

    /** @} */ // end of ovrtx_error

#ifdef __cplusplus
}
#endif

#endif /* OVRTX_H */

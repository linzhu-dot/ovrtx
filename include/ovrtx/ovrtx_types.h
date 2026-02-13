/* Copyright (c) 2025-2026, NVIDIA CORPORATION. All rights reserved.
 *
 * NVIDIA CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA CORPORATION is strictly prohibited.
 */

#ifndef OVRTX_TYPES_H
#define OVRTX_TYPES_H

#define OVRTX_VERSION_MAJOR 0
#define OVRTX_VERSION_MINOR 1
#define OVRTX_VERSION_PATCH 0

#include <stdint.h>
#include <stdbool.h>
#include <stddef.h>

#include "../ovx/dlpack/dlpack.h"
#include "../ovx/types.h"
#include "../ovx/path_dictionary/path_dictionary.h"

#ifdef __cplusplus
extern "C"
{
#endif

    /** @defgroup ovrtx_core_types Core renderer types
     *  @{
     */

    typedef struct ovrtx_renderer_t ovrtx_renderer_t;

    /** @} */ // end of ovrtx_core_types

    /** @defgroup ovrtx_handle_types Handle types
     *  @{
     */

    /** Handle representing a USD stage. */
    typedef uint64_t ovrtx_usd_handle_t;
    /** Handle representing an event. */
    typedef uint64_t ovrtx_event_handle_t;
    /** Handle representing a persistent attribute binding. */
    typedef uint64_t ovrtx_attribute_binding_handle_t;
    /** Handle representing a resource mapping that can be used to unmap it. */
    typedef uint64_t ovrtx_map_handle_t;
    /** Handle to the result of a @ref ovrtx_step() operation. */
    typedef uint64_t ovrtx_step_result_handle_t;
    /** Handle to a rendered output. */
    typedef uint64_t ovrtx_rendered_output_handle_t;
    /** Handle to the mapping of a rendered output that can be used to unmap it. */
    typedef uint64_t ovrtx_rendered_output_map_handle_t;
    /** Identifier of a particular asynchronous operation such as @ref ovrtx_add_usd() that can be used to poll or wait. */
    typedef uint64_t ovrtx_op_id_t;

    /**
    * Sentinel value representing an invalid/null ovrtx handle.
    * Valid handles non-zero, so 0 is reserved to indicate "no handle" or "invalid handle".
    * Zero initialization of structures containing handles is valid API usage to indicate
    * that no valid handle is provided.
    */
    #define OVRTX_INVALID_HANDLE 0

    /** @} */ // end of ovrtx_handle_types

    /** @defgroup ovrtx_result_types Result and status types
     *  @{
     */

    /**
     * Return status from a synchronous function call.
     */
    typedef enum
    {
        OVRTX_API_SUCCESS = 0, /**< The function completed successfully. */
        OVRTX_API_ERROR = 1, /**< The function generated an error. The associated error message can be queried with 
                                  @ref ovrtx_get_last_error(). */
        OVRTX_API_TIMEOUT = 2, /**< The timeout passed to the function was reached. */
    } ovrtx_api_status_t;

    /**
     * Result from a synchronous function call. The status of the call can be checked with the @ref ovrtx_result_t::status member.
     */
    typedef struct
    {
        ovrtx_api_status_t status; /**< Status of the call. */
    } ovrtx_result_t;

    /**
     * @brief Result from an aynschronous function call.
     *
     * The immediate status of the call can be checked with the @ref ovrtx_enqueue_result_t::status field, and @ref ovrtx_enqueue_result_t::op_index can be used to poll 
     * or wait on the completion of the asynchronous work.
     */
    typedef struct
    {
        ovrtx_api_status_t status; /**< Status of the call */
        ovrtx_op_id_t op_index; /**< Operation identifier that can be used to poll or wait on the completion of the 
                                     enqueued work */
    } ovrtx_enqueue_result_t;

    typedef enum
    {
        OVRTX_EVENT_PENDING = 0,
        OVRTX_EVENT_COMPLETED = 1,
        OVRTX_EVENT_FAILURE = 2,
    } ovrtx_event_status_t;

    /** @} */ // end of ovrtx_result_types

    /** @defgroup ovrtx_op_wait_types Per-operation wait result types
     *  @{
     */

    /** Result of waiting on an @ref ovrtx_op_id_t with @ref ovrtx_wait_op(). */
    typedef struct
    {
        /** List of operation ids that errored since last wait call. */
        ovrtx_op_id_t* error_op_ids;
        size_t num_error_ops;

        /** The lowest operation id that is still pending, or 0 if all operations are complete. */
        ovrtx_op_id_t lowest_pending_op_id;
    } ovrtx_op_wait_result_t;

    /** @} */ // end of ovrtx_op_wait_types

    /** @defgroup ovrtx_sync_types Stream synchronization types
     *  @{
     */

    /** Represents a timeout duration in nanoseconds. */
    typedef struct
    {
        uint64_t time_out_ns;
    } ovrtx_timeout_t;

    /** Represents a CUDA event to wait for on a particular stream. */
    typedef struct
    {
        uintptr_t stream; /**< Cuda stream to synchronize to. 0 = no synchronization. 1 = default cuda stream, >1 specific stream */
        uintptr_t wait_event; /**< Event to wait on before operation (0 = none) */
    } ovrtx_cuda_sync_t;

    typedef struct
    {
        int32_t device_type; /**< DLDevice device_type */
        int32_t device_id;   /**< DLDevice device_id */
    } ovrtx_device_t;

    typedef struct
    {
        ovrtx_device_t device; /**< device to create the synchronization event on */
    } ovrtx_event_description_t;

    /** @} */ // end of ovrtx_sync_types


    /** @defgroup ovrtx_attribute_types Stage attribute writer types
     *  @{
     */

    typedef enum
    {
        OVRTX_DIRTY_MASK_REPLACE = 0, /**< Replace consumer dirty mask */
        OVRTX_DIRTY_MASK_OR = 1, /**< OR masks together */
        OVRTX_DIRTY_MASK_AND = 2 /**< AND masks together */
    } ovrtx_write_bits_t;

    typedef enum
    {
        OVRTX_DATA_ACCESS_ASYNC = 0, /**< Accesses the inputs asynchronously as part of the stream execution, 
                                        so the input lifetime must remain valid until completion (use ovrtx_wait_op). */
        OVRTX_DATA_ACCESS_SYNC = 1, /**< Copies all input data synchronously, 
                                        so the data is no longer accessed when the call returns. */
    } ovrtx_data_access_t;

    /** A list of paths to prims in the runtime stage. */
    typedef struct ovrtx_prim_list_t
    {
        const ovx_string_t* prim_paths;
        size_t num_paths;
    } ovrtx_prim_list_t;

    typedef struct
    {
        int32_t device_type; /**< device_type from DLDevice */
        int32_t device_id;   /**< device_id from DLDevice */
    } ovrtx_mapping_desc_t;

    /** Describes how to handle attempts to write to paths in the runtime stage that do not exist. */
    typedef enum ovrtx_binding_prim_mode_t
    {
        OVRTX_BINDING_PRIM_MODE_EXISTING_ONLY = 0, /**< Only existing attributes on existingprims are written to, prims that do not exist are ignored */
        OVRTX_BINDING_PRIM_MODE_MUST_EXIST = 1, /**< All attributes and prims must exist to write any data successfully */
        OVRTX_BINDING_PRIM_MODE_CREATE_NEW = 2, /**< New attribute/prim pairs are created when written to */
    } ovrtx_binding_prim_mode_t;

    /** Used to differentiate the intended usage of a given attribute type. 
     *  
     * For example, an attribute with 3-float elements could be interpreted as a vector or as an RGB color. This enum
     * differentiates between them.
     */
    typedef enum ovrtx_attribute_semantic_t
    {
        OVRTX_SEMANTIC_NONE = 0, 
        OVRTX_SEMANTIC_TRANSFORM_4x4 = 1, /**< Transform of a prim expressed as column-major 4x4 matrix of double (kDLFloat, 64, 16) */
        OVRTX_SEMANTIC_TRANSFORM_POS3d_ROT4f_SCALE3f = 2, /**< Transform of a prim expressed as 3xdouble position, 4xfloat rotation and 3xfloat scale (kDLUInt, 8, 56) */
        OVRTX_SEMANTIC_TRANSFORM_POS3d_ROT3x3f = 3, /**< Transform of a prim expressed as 3xdouble position, 3x3float rotation matrix (kDLUInt, 8, 64) */
        OVRTX_SEMANTIC_PATH_STRING = 4, /**< Prim paths expressed as ovx_string_t (kDLUInt, 128, 1). The strings must be valid for the duration of the write_attribute call. Only synchronous data access is supported.*/
        OVRTX_SEMANTIC_TOKEN_STRING = 5,  /**< String token expressed as ovx_string_t (kDLUInt, 128, 1). The strings must be valid for the duration of the write_attribute call. Only synchronous data access is supported.*/
        OVRTX_SEMANTIC_COLOR_RGBA4b = 6, /**< A color expressed as 4 bytes (kDLUInt, 8, 4) */
        OVRTX_SEMANTIC_COLOR_RGB3f = 7, /**< A color expressed as 3 floats (kDLFloat, 32, 3) */
    }ovrtx_attribute_semantic_t;

    /** Describes the type of an attribute to be written to the runtime stage. */
    typedef struct  
    {
        DLDataType dtype; /**< Data type */
        bool is_array;  /**< Whether this attribute is an array attribute. Array attributes are variable length per attribute */
        ovrtx_attribute_semantic_t semantic; /**< The interpretation of this array data in a USD stage */
    }ovrtx_attribute_type_t;

    /** Flags giving hints to the renderer about the expected use of a binding. */
    typedef enum ovrtx_binding_flag_t
    {
        OVRTX_BINDING_FLAG_NONE = 0,
        /**< Indicates that the renderer should optimize it's underlying data structures for frequent high volume writes 
             to this binding. The last created binding with this flag will be prioritized. */
        OVRTX_BINDING_FLAG_OPTIMIZE = 1 << 0,  
    }ovrtx_binding_flag_t;

    /** Describes a binding to an attribute on a list of prims so that they can be written to. */
    typedef struct
    {
        ovrtx_prim_list_t prim_list; /**< Explicit list of prims when no handle provided */
        ovx_primpath_list_t prims_list_handle; /**< Handle to a persistent prim list */

        ovx_string_or_token_t attribute_name; /**< Name of the attribute*/
        ovrtx_attribute_type_t attribute_type; /**< Type of the attribute being bound */

        ovrtx_binding_prim_mode_t prim_mode; /**< Mode to determine how prims are handled */
        ovrtx_binding_flag_t flags; /**< Additional flags to control the binding behavior */
    } ovrtx_binding_desc_t;

    /** Represents either an ovrtx_binding_desc_t or an ovrtx_attribute_binding_handle_t allowing either to be passed to 
     *  @ref ovrtx_write_attribute() and @ref ovrtx_map_attribute().
     *  
     * The use of persistent bindings allows for more optimal writes in the renderer when an attribute will be written to 
     * repeatedly.
     *
     * If @ref ovrtx_binding_desc_or_handle_t::binding_handle is non-zero then it will be used, otherwise 
     * @ref ovrtx_binding_desc_or_handle_t::binding_desc will be used.
     */
    typedef struct
    {
        ovrtx_binding_desc_t binding_desc; /**< binding description if no handle provided */
        ovrtx_attribute_binding_handle_t binding_handle; /**< handle to a persistent binding */
    } ovrtx_binding_desc_or_handle_t;

    /** Represents a USD stage or layer to be used as input to @ref ovrtx_add_usd() */
    typedef struct ovrtx_usd_input_t
    {
        ovx_string_t usd_file_path; /**< Path to the USD file to add */
        uint64_t usd_stage_id; /**< Stage ID of a USD runtime stage to add */
        ovx_string_t usd_layer_content; /**< Usda content of the layer to add */
    } ovrtx_usd_input_t;

    typedef struct
    {
        /** @name Input buffers as an array of tensors. 
         *
         * For array attributes the array length must 
         * match the number of prims in the attribute binding otherwise it must be 1. The tensor
         * object (not the data) are copied during the call and don't have to persist for longer than
         * the duration of the call. 
         * @{
         */
        DLTensor* tensors; /**< Array of tensors. */
        uint64_t tensor_count; /**< Number of tensors in the array. */
        /** @} */

        /** @name Dirty bit tracking for selective updates. 
          * The dirty bit array is copied and must not persist longer than the call. 
          * @{
          */
        uint8_t* dirty_bits; /**< Bitvector: 1 bit per prim for dirty tracking */
        size_t dirty_bits_size; /**< Size of dirty_bits array: (prim_count + 7) / 8 */
        /** @} */

        /** @name Synchronization
         * @{
         */
        /** Optional synchronization hint used to guard access to the input data inside the renderer */
        ovrtx_cuda_sync_t access_cuda_sync;
        /** Optional synchronization hint used to signal that the input data is no longer accessed by the renderer */
        ovrtx_cuda_sync_t done_cuda_sync;
        /** @} */
    } ovrtx_input_buffer_t;

    /** Output DLTensor generated for a particular frame, product and var. */
    typedef struct
    {
        DLTensor dl; /**< Zero-copy tensor (required) */
        ovrtx_cuda_sync_t cuda_sync; /**< Optional CUDA synchronization hints associated with this buffer */
    } ovrtx_output_buffer_t;

    /** Mapped attribute that can be written to until unmapped. */
    typedef struct
    {
        ovrtx_map_handle_t map_handle; /**< Map handle for unmap operation */
        DLTensor dl; /**< Mapped memory as tensor, valid until unmap */
    } ovrtx_attribute_mapping_t;

    /** @} */ // end of ovrtx_attribute_types

    /** @defgroup ovrtx_sensor_types Sensor simulation types
     *  @{
     */

    /** Set of RenderProducts that will be stepped. */
    typedef struct ovrtx_render_product_set_t
    {
        const ovx_string_t* render_products; /**< Array of paths to RenderProduct prims in the stage. */
        size_t num_render_products; /**< Number of paths in the array. */
    } ovrtx_render_product_set_t;

    /** Enum representing the status of an asynchronous operation. */
    typedef enum
    {
        OVRTX_RENDERER_EVENT_PENDING = 0, /**< Operation still in progress */
        OVRTX_RENDERER_EVENT_COMPLETED = 1, /**< Operation completed successfully */
        OVRTX_RENDERER_EVENT_FAILED = 2 /**< Operation failed with error */
    } ovrtx_renderer_event_status_t;

    /** Name an associated handle of a particular RenderVar's output in a RenderProduct output.  */
    typedef struct
    {
        ovx_string_t render_var_name; /**< Name of the associated render var */
        ovrtx_rendered_output_handle_t output_handle; /**< Handle to the rendered output */
    } ovrtx_render_product_render_var_output_t;

    /** Output of a particular RenderProduct for a particular frame. 
     *
     * May contain one or more RenderVar outputs in @ref ovrtx_render_product_frame_output_t::output_render_vars which may
     * each be mapped to get access to the output data using @ref ovrtx_map_rendered_output().
     */
    typedef struct
    {
        double frame_start_time; /**< Sensor time (based on step(delta_time) history) when the sensor simulation for this frame started */
        double frame_end_time; /**< Sensor time (based on step(delta_time) history) when the sensor simulation for this frame ended */
        ovrtx_render_product_render_var_output_t* output_render_vars; /**< Pointer to array of render var outputs */
        size_t render_var_count; /**< Number of render var outputs for this render product frame */
    } ovrtx_render_product_frame_output_t;

    /** The output of a particular RenderProduct for a particular @ref ovrtx_step() operation. 
     */
    typedef struct
    {
        ovx_string_t render_product_path; /**< Path of the render product that produced this output */
        float output_frames_produced; /**< Decimal value representing the amount of frames produced */
        ovrtx_render_product_frame_output_t* output_frames; /**< pointer to array of output frames */
        size_t output_frame_count; /**< Number of frames for this render product */
    } ovrtx_render_product_output_t;

    /**
     * The set of RenderProduct outputs for a @ref ovrtx_step() operation.
     * 
     * Depending on the sensor configuration, each @ref ovrtx_step() may produce zero or more frames for each RenderProduct
     * in the @ref ovrtx_render_product_set_t passed to @ref ovrtx_step().
     */
    typedef struct
    {
        ovrtx_event_status_t status; /**< Current operation status */
        ovx_string_t error_message; /**< Error description if failed, NULL if succeeded */
        double simulation_start_time; /**< Sensor time (based on step(delta_time) history) when the step which produced this output started */
        double simulation_end_time; /**< Sensor time (based on step(delta_time) history) when the step which produced this output ended */
        ovrtx_render_product_output_t* outputs; /**< Array of outputs returned */
        size_t output_count; /**< Number of outputs in the array */
        double start_time; /**< Sensor time (based on step(delta_time) history) when the sensor simulation for this step started */
        double end_time; /**< Sensor time (based on step(delta_time) history) when the sensor simulation for this step ended */
    } ovrtx_render_product_set_outputs_t;


    /** Specifies which device (CPU or GPU) should be used to map a given output. */
    typedef enum
    {
        /**< Provide the data in whatever format is most efficient. */
        OVRTX_MAP_DEVICE_TYPE_DEFAULT = 0,
        /**< Read back output from GPU to CPU. This will incur synchronization and copy. */
        OVRTX_MAP_DEVICE_TYPE_CPU = 1,
        /**< Raw CUDA device memory as dltensor. This will likely incur an additional copy when used for image outputs. */
        OVRTX_MAP_DEVICE_TYPE_CUDA = 2, 
        /**< CUDA array represented in dltensor as kDLCUDA / kDLOpaqueHandle. This saves a copy for image outputs, but is not supported for other outputs. */
        OVRTX_MAP_DEVICE_TYPE_CUDA_ARRAY = 3,
    } ovrtx_map_device_type_t;

    /** Description of the device and synchronization for mapping an output to be passed to @ref ovrtx_map_rendered_output(). */
    typedef struct
    {
        ovrtx_map_device_type_t device_type; /**< Device type of the output */
        uintptr_t sync_stream; /**< CUDA stream to synchronize production of the output with. 
                                Providing a stream here means that after the map call returns the output
                                data can immediately be accessed on a cuda stream that is synchronized to the provided stream.
                                 0 = no synchronization. 1 = default cuda stream, >1 specific stream */
        
    } ovrtx_map_output_description_t;

    /** The output of a particular RenderVar for a particular RenderProduct on a particular frame. */
    typedef struct
    {
        ovrtx_event_status_t status; /**< Current operation status */
        ovx_string_t error_message; /**< Error description if failed, NULL if succeeded */
        ovrtx_rendered_output_map_handle_t map_handle; /**< Handle to use for unmap */
        ovx_string_t name; /**< Name of the output (e.g., "rgb", "depth") */
        ovrtx_output_buffer_t buffer; /**< Buffer containing the rendered data */
    } ovrtx_rendered_output_t;

    /** Type of a value contained in @ref ovrtx_renderer_config_value_t */
    typedef enum ovrtx_renderer_config_value_type_t
    {
        OVRTX_CONFIG_VALUE_BOOL,
        OVRTX_CONFIG_VALUE_INT64,
        OVRTX_CONFIG_VALUE_UINT64,
        OVRTX_CONFIG_VALUE_DOUBLE,
        OVRTX_CONFIG_VALUE_STRING,
        OVRTX_CONFIG_VALUE_BLOB,
    } ovrtx_renderer_config_value_type_t;

    /** A value to be passed as an entry in the @ref ovrtx_config_t configuration dictionary. */
    typedef struct
    {
        ovrtx_renderer_config_value_type_t type;
        union
        {
            bool bool_value;
            int64_t int_value;
            uint64_t uint_value;
            double double_value;
            ovx_string_t string_value;
            struct
            {
                const void* data;
                size_t size;
            } blob_value;
        };
    } ovrtx_renderer_config_value_t;

    /** An key/value entry in the @ref ovrtx_config_t configuration dictionary. */
    typedef struct
    {
        ovx_string_t key;
        ovrtx_renderer_config_value_t value;
    } ovrtx_renderer_config_entry_t;

    /** A dictionary of key/value pairs that can be passed to @ref ovrtx_initialize() or @ref ovrtx_create_renderer(). */
    typedef struct
    {
        const ovrtx_renderer_config_entry_t* entries;
        size_t entry_count;
    } ovrtx_config_t;

    /** @ref ovrtx_timeout_t constant for infinity (i.e. block indefinitely) */
    static const ovrtx_timeout_t ovrtx_timeout_infinite = { (size_t)-1 }; /* .time_out_ns = would require C++20 */

    /** @} */ // end of ovrtx_sensor_types

#ifdef __cplusplus
}
#endif

#endif /* OVRTX_TYPES_H */

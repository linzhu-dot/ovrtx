/* Copyright (c) 2025-2026, NVIDIA CORPORATION. All rights reserved.
 *
 * NVIDIA CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA CORPORATION is strictly prohibited.
 */

/** @file ovrtx_config.h */

#ifndef OVRTX_CONFIG_H
#define OVRTX_CONFIG_H

#include "ovrtx_types.h"

#ifdef __cplusplus
extern "C"
{
#endif

/** Path to the OVRTX "binary package root". Used by the loader to find required runtime files. */
#define OVRTX_CONFIG_KEY_BINARY_PACKAGE_ROOT_PATH "binary_package_root_path"
/** If true, operations enqueued into the renderer stream execute synchronously (enqueue blocks until complete). */
#define OVRTX_CONFIG_KEY_SYNC_MODE "sync_mode"
/** If true, enables internal profiling in the renderer. */
#define OVRTX_CONFIG_KEY_ENABLE_PROFILING "enable_profiling"
/** If true, transform propagation during rendering uses GPU world transform updates. */
#define OVRTX_CONFIG_KEY_READ_GPU_TRANSFORMS "read_gpu_transforms"
/** If true, the renderer will output partial frames for incremental sensors. */
#define OVRTX_CONFIG_KEY_OUTPUT_PARTIAL_FRAMES "output_partial_frames"
/** Path to the log file for carb logging. */
#define OVRTX_CONFIG_KEY_LOG_FILE_PATH "log_file_path"
/** Log level for carb logging. */
#define OVRTX_CONFIG_KEY_LOG_LEVEL "log_level"


/**
 * Build a configuration entry for a boolean value.
 * @param key Configuration key.
 * @param value Stored by value in the entry.
 */
static inline ovrtx_renderer_config_entry_t ovrtx_config_entry_bool(ovx_string_t key, bool value)
{
    ovrtx_renderer_config_entry_t entry;
    entry.key = key;
    entry.value.type = OVRTX_CONFIG_VALUE_BOOL;
    entry.value.bool_value = value;
    return entry;
}

/**
 * Build a configuration entry for a 64-bit integer value.
 * @param key Configuration key.
 * @param value Stored by value in the entry.
 */
static inline ovrtx_renderer_config_entry_t ovrtx_config_entry_int64(ovx_string_t key, int64_t value)
{
    ovrtx_renderer_config_entry_t entry;
    entry.key = key;
    entry.value.type = OVRTX_CONFIG_VALUE_INT64;
    entry.value.int_value = value;
    return entry;
}

/**
 * Build a configuration entry for a 64-bit unsigned integer value.
 * @param key Configuration key.
 * @param value Stored by value in the entry.
 */
static inline ovrtx_renderer_config_entry_t ovrtx_config_entry_uint64(ovx_string_t key, uint64_t value)
{
    ovrtx_renderer_config_entry_t entry;
    entry.key = key;
    entry.value.type = OVRTX_CONFIG_VALUE_UINT64;
    entry.value.uint_value = value;
    return entry;
}

/**
 * Build a configuration entry for a double value.
 * @param key Configuration key.
 * @param value Stored by value in the entry.
 */
static inline ovrtx_renderer_config_entry_t ovrtx_config_entry_double(ovx_string_t key, double value)
{
    ovrtx_renderer_config_entry_t entry;
    entry.key = key;
    entry.value.type = OVRTX_CONFIG_VALUE_DOUBLE;
    entry.value.double_value = value;
    return entry;
}

/**
 * Build a configuration entry for a string value.
 * @param key Configuration key.
 * @param value String value. value.ptr must remain valid until the config is consumed.
 */
static inline ovrtx_renderer_config_entry_t ovrtx_config_entry_string(ovx_string_t key, ovx_string_t value)
{
    ovrtx_renderer_config_entry_t entry;
    entry.key = key;
    entry.value.type = OVRTX_CONFIG_VALUE_STRING;
    entry.value.string_value = value;
    return entry;
}

/**
 * Build a configuration entry for a blob value.
 * @param key Configuration key.
 * @param data Blob value. data must remain valid until the config is consumed.
 * @param size Size of the blob in bytes.
 */
static inline ovrtx_renderer_config_entry_t ovrtx_config_entry_blob(ovx_string_t key, const void* data, size_t size)
{
    ovrtx_renderer_config_entry_t entry;
    entry.key = key;
    entry.value.type = OVRTX_CONFIG_VALUE_BLOB;
    entry.value.blob_value.data = data;
    entry.value.blob_value.size = size;
    return entry;
}

/**
 * Configure the OVRTX "binary package root" directory.
 *
 * This is used by the loader to locate the renderer shared library and required runtime resources.
 * If not provided, the loader chooses a default based on the loader's location.
 *
 * @param path The root directory path. path.ptr must remain valid until the API call that consumes the config returns.
 */
static inline ovrtx_renderer_config_entry_t ovrtx_config_entry_binary_package_root_path(ovx_string_t path)
{
    ovx_string_t key;
    key.ptr = OVRTX_CONFIG_KEY_BINARY_PACKAGE_ROOT_PATH;
    key.length = sizeof(OVRTX_CONFIG_KEY_BINARY_PACKAGE_ROOT_PATH) - 1;
    return ovrtx_config_entry_string(key, path);
}

/**
 * Configure synchronous stream execution.
 *
 * When enabled, operations that are normally enqueued for stream-ordered execution will block
 * the calling thread until the operation has completed. Any errors will be returned immediately.
 */
static inline ovrtx_renderer_config_entry_t ovrtx_config_entry_sync_mode(bool sync_mode)
{
    ovx_string_t key;
    key.ptr = OVRTX_CONFIG_KEY_SYNC_MODE;
    key.length = sizeof(OVRTX_CONFIG_KEY_SYNC_MODE) - 1;
    return ovrtx_config_entry_bool(key, sync_mode);
}

/**
 * Configure internal profiling.
 *
 * When enabled, the renderer collects additional profiling data. This can add overhead.
 */
static inline ovrtx_renderer_config_entry_t ovrtx_config_entry_enable_profiling(bool enable_profiling)
{
    ovx_string_t key;
    key.ptr = OVRTX_CONFIG_KEY_ENABLE_PROFILING;
    key.length = sizeof(OVRTX_CONFIG_KEY_ENABLE_PROFILING) - 1;
    return ovrtx_config_entry_bool(key, enable_profiling);
}

/**
 * Configure transform propagation mode.
 *
 * When enabled, the renderer uses GPU world transform propagation during rendering.
 */
static inline ovrtx_renderer_config_entry_t ovrtx_config_entry_read_gpu_transforms(bool read_gpu_transforms)
{
    ovx_string_t key;
    key.ptr = OVRTX_CONFIG_KEY_READ_GPU_TRANSFORMS;
    key.length = sizeof(OVRTX_CONFIG_KEY_READ_GPU_TRANSFORMS) - 1;
    return ovrtx_config_entry_bool(key, read_gpu_transforms);
}

/**
 * Configure partial frame output.
 *
 * When enabled, the renderer will output partial frames for incremental sensors.
 * When disabled, the renderer will only output full frames for incremental sensors.
 */
static inline ovrtx_renderer_config_entry_t ovrtx_config_entry_output_partial_frames(bool partial_frames)
{
    ovx_string_t key;
    key.ptr = OVRTX_CONFIG_KEY_OUTPUT_PARTIAL_FRAMES;
    key.length = sizeof(OVRTX_CONFIG_KEY_OUTPUT_PARTIAL_FRAMES) - 1;
    return ovrtx_config_entry_bool(key, partial_frames);
}

/**
 * Configure log file path for carb logging.
 *
 * This setting should be passed to ovrtx_initialize(). The log file is created when the first
 * renderer is created via ovrtx_create_renderer().
 *
 * The path may contain the ${start_timestamp} token which will be replaced with the startup
 * timestamp in YYYYMMDD_HHMMSS format (e.g., "myapp_${start_timestamp}.log").
 *
 * If not specified, defaults to: <app_directory>/ovrtx_${start_timestamp}.log
 *
 * @param path Log file path. path.ptr must remain valid until the API call that consumes the config returns.
 */
static inline ovrtx_renderer_config_entry_t ovrtx_config_entry_log_file_path(ovx_string_t path)
{
    ovx_string_t key;
    key.ptr = OVRTX_CONFIG_KEY_LOG_FILE_PATH;
    key.length = sizeof(OVRTX_CONFIG_KEY_LOG_FILE_PATH) - 1;
    return ovrtx_config_entry_string(key, path);
}

/**
 * Configure carb logging level (e.g. "verbose", "info", "warn", "error").
 *
 * This setting should be passed to ovrtx_initialize(). The log level is applied when the first
 * renderer is created via ovrtx_create_renderer().
 *
 * @param level Log level string. level.ptr must remain valid until the API call that consumes the config returns.
 */
static inline ovrtx_renderer_config_entry_t ovrtx_config_entry_log_level(ovx_string_t level)
{
    ovx_string_t key;
    key.ptr = OVRTX_CONFIG_KEY_LOG_LEVEL;
    key.length = sizeof(OVRTX_CONFIG_KEY_LOG_LEVEL) - 1;
    return ovrtx_config_entry_string(key, level);
}

#ifdef __cplusplus
}
#endif

#endif /* OVRTX_CONFIG_H */



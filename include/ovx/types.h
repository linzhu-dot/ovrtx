/* Copyright (c) 2025-2026, NVIDIA CORPORATION. All rights reserved.
 *
 * NVIDIA CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA CORPORATION is strictly prohibited.
 */

#ifndef OVX_TYPES_H
#define OVX_TYPES_H

#include <stdint.h>
#include <stddef.h>

/* Null-terminated string with explicit length */
typedef struct ovx_string_t
{
    const char* ptr;
    size_t length;
} ovx_string_t;

typedef uint64_t ovx_token_t;
typedef uint64_t ovx_primpath_t;
typedef uint64_t ovx_primpath_list_t;

typedef enum
{
    OVX_API_SUCCESS = 0,
    OVX_API_ERROR = 1,
} ovx_api_status_t;

typedef struct
{
    ovx_api_status_t status;
    ovx_string_t error;
} ovx_api_result_t;

typedef struct ovx_string_or_token_t
{
    ovx_token_t token; /* Optional token handle within the path_dictionary_interface */
    ovx_string_t string;
} ovx_string_or_token_t;

typedef struct ovx_string_or_prim_path_t
{
    ovx_primpath_t path; /* Optional prim path handle within the path_dictionary_interface */
    ovx_string_t string;
} ovx_string_or_prim_path_t;

#endif /* OVX_TYPES_H */

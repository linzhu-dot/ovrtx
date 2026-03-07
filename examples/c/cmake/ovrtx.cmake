# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

# ovrtx.cmake - Fetch and configure ovrtx library
#
# Usage:
#   list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_LIST_DIR}/../cmake")
#   include(ovrtx)
#   ovrtx_fetch()
#   
#   add_executable(myapp main.cpp)
#   target_link_libraries(myapp PRIVATE ovrtx::ovrtx)
#   ovrtx_setup_runtime(myapp)

# Capture this file's directory at parse time (before macro expansion)
# CMAKE_CURRENT_LIST_DIR inside a macro would refer to the caller's directory
set(_OVRTX_CMAKE_DIR "${CMAKE_CURRENT_LIST_DIR}")

# Fetch ovrtx package from NVIDIA artifactory
macro(ovrtx_fetch)
    find_package(ovrtx QUIET)
    if (ovrtx_FOUND)
        message(STATUS "found ovrtx at: ${ovrtx_DIR}")
    else()
        set(FETCHCONTENT_QUIET FALSE)
        
        # Override FetchContent's base directory to share large deps among examples.
        # Uses the directory where ovrtx.cmake lives, ensuring all examples share the same _deps.
        # If copying this project to your own workspace, delete this line or override it.
        if(NOT DEFINED CACHE{FETCHCONTENT_BASE_DIR})
            set(FETCHCONTENT_BASE_DIR "${_OVRTX_CMAKE_DIR}/_deps" CACHE PATH "Shared FetchContent directory")
        endif()

        # Platform-specific package selection
        if(CMAKE_SYSTEM_NAME STREQUAL "Windows")
            set(OVRTX_PACKAGE_SYSTEM "windows-x86_64")
            set(OVRTX_HASH "aa9aa3768926adcaa4066cd84b10f28b05d579dbfdaed5260a8fa63b8a65aef1")
        elseif(CMAKE_SYSTEM_NAME STREQUAL "Linux")
            if (CMAKE_SYSTEM_PROCESSOR STREQUAL "aarch64")
                set(OVRTX_PACKAGE_SYSTEM "manylinux_2_35_aarch64")
                set(OVRTX_HASH "b2a6ac57b70b2d039cce06a0b99da07e1273494edce73bf0314bfdfb20d34a6d")
            elseif(CMAKE_SYSTEM_PROCESSOR STREQUAL "x86_64")
                set(OVRTX_PACKAGE_SYSTEM "manylinux_2_35_x86_64")
                set(OVRTX_HASH "d5bab8ddd375194dd378499230828bcc0718b69197628f457c84bdcb1f1d3213")
            else()
                message(FATAL_ERROR "Unsupported system: ${CMAKE_SYSTEM_NAME} ${CMAKE_SYSTEM_PROCESSOR}")
            endif()
        else()
            message(FATAL_ERROR "Unsupported system: ${CMAKE_SYSTEM_NAME} ${CMAKE_SYSTEM_PROCESSOR}")
        endif()

        include(FetchContent)

        FetchContent_Declare(
            ovrtx
            URL "https://github.com/NVIDIA-Omniverse/ovrtx/releases/download/v0.2.0/ovrtx@0.2.0.280040.ac9618b8.${OVRTX_PACKAGE_SYSTEM}.zip"
            URL_HASH SHA256=${OVRTX_HASH}
            DOWNLOAD_EXTRACT_TIMESTAMP TRUE
        )


        FetchContent_MakeAvailable(ovrtx)
        
        # Make ovrtx findable by find_package
        list(APPEND CMAKE_PREFIX_PATH ${ovrtx_SOURCE_DIR})

        find_package(ovrtx REQUIRED)

    endif()
endmacro()

# Setup runtime dependencies for a target (DLL copying, junctions, rpath)
function(ovrtx_setup_runtime TARGET_NAME)
    if(CMAKE_SYSTEM_NAME STREQUAL "Windows")
        # Copy ovrtx DLL to build directory
        add_custom_command(TARGET ${TARGET_NAME} POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy_if_different
                "${OVRTX_BINARY_DIR}/ovrtx-dynamic.dll"
                "$<TARGET_FILE_DIR:${TARGET_NAME}>"
            COMMENT "Copying ovrtx-dynamic.dll to build directory"
        )
        
        # Create junctions for ovrtx runtime files (plugins, etc.)
        # Use PowerShell's New-Item which handles path separators correctly
        set(OVRTX_RUNTIME_DIRS cache library libs mdl plugins rendering-data usd_plugins)
        message(STATUS "Creating ovrtx junctions for runtime files if needed")
        foreach(DIR ${OVRTX_RUNTIME_DIRS})
            add_custom_command(TARGET ${TARGET_NAME} POST_BUILD
                COMMAND ${CMAKE_COMMAND} -E env powershell -NoProfile -Command
                    "if (-not (Test-Path '$<TARGET_FILE_DIR:${TARGET_NAME}>/${DIR}')) { New-Item -ItemType Junction -Path '$<TARGET_FILE_DIR:${TARGET_NAME}>/${DIR}' -Target '${OVRTX_BINARY_DIR}/${DIR}' | Out-Null }"
            )
        endforeach()
    else()
        # Set rpath so the executable can find libovrtx-dynamic.so at runtime
        set_target_properties(${TARGET_NAME} PROPERTIES
            BUILD_RPATH "${OVRTX_BINARY_DIR}"
            INSTALL_RPATH "${OVRTX_BINARY_DIR}"
        )
    endif()
endfunction()

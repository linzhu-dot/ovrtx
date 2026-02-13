.. SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
.. SPDX-License-Identifier: LicenseRef-NvidiaProprietary
..
.. NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
.. property and proprietary rights in and to this material, related
.. documentation and any modifications thereto. Any use, reproduction,
.. disclosure or distribution of this material and related documentation
.. without an express license agreement from NVIDIA CORPORATION or
.. its affiliates is strictly prohibited.

C API Reference
===============

The C API provides low-level access to the ovrtx rendering library.

.. contents:: Contents
   :local:
   :depth: 2

Overview
--------

All operations return a status code indicating success or failure:

- **OVRTX_API_SUCCESS**: The operation completed successfully
- **OVRTX_API_ERROR**: The operation failed (use :c:func:`ovrtx_get_last_error` for details)
- **OVRTX_API_TIMEOUT**: The operation timed out

Many operations are stream-ordered and execute asynchronously. They return handles that can be used in subsequent operations, but the actual effects may not be produced until stream execution completes.

Known Limitations
-----------------

In the current version, if you wish to use OpenUSD in the same process as ovrtx, you must use the same version of OpenUSD as the one used to build ovrtx. This means that you must use `OpenUSD v25.11 <https://github.com/PixarAnimationStudios/OpenUSD/tree/v25.11>`_, built as non-monolithic libraries with Python support enabled (the default). You must also use oneTBB v2021.13.0 or later when building OpenUSD.

Functions
---------

Creation and Destruction
^^^^^^^^^^^^^^^^^^^^^^^^

.. doxygengroup:: ovrtx_creation
   :project: ovrtx
   :content-only:

Stage Building
^^^^^^^^^^^^^^

.. doxygengroup:: ovrtx_stage_building
   :project: ovrtx
   :content-only:

Attribute Operations
^^^^^^^^^^^^^^^^^^^^

.. doxygengroup:: ovrtx_attribute_write
   :project: ovrtx
   :content-only:

Sensor Simulation
^^^^^^^^^^^^^^^^^

.. doxygengroup:: ovrtx_sensor_simulation
   :project: ovrtx
   :content-only:

Stream Operations
^^^^^^^^^^^^^^^^^

.. doxygengroup:: ovrtx_stream
   :project: ovrtx
   :content-only:

Extensions
^^^^^^^^^^

.. doxygengroup:: ovrtx_extension
   :project: ovrtx
   :content-only:

Error Handling
^^^^^^^^^^^^^^

.. doxygengroup:: ovrtx_error
   :project: ovrtx
   :content-only:

Types
-----

Core Types
^^^^^^^^^^

.. doxygengroup:: ovrtx_core_types
   :project: ovrtx
   :content-only:

Handle Types
^^^^^^^^^^^^

.. doxygengroup:: ovrtx_handle_types
   :project: ovrtx
   :content-only:

Result Types
^^^^^^^^^^^^

.. doxygengroup:: ovrtx_result_types
   :project: ovrtx
   :content-only:

Operation Wait Types
^^^^^^^^^^^^^^^^^^^^

.. doxygengroup:: ovrtx_op_wait_types
   :project: ovrtx
   :content-only:

Synchronization Types
^^^^^^^^^^^^^^^^^^^^^

.. doxygengroup:: ovrtx_sync_types
   :project: ovrtx
   :content-only:

User Task Types
^^^^^^^^^^^^^^^

.. doxygengroup:: ovrtx_user_task_types
   :project: ovrtx
   :content-only:

Attribute Types
^^^^^^^^^^^^^^^

.. doxygengroup:: ovrtx_attribute_types
   :project: ovrtx
   :content-only:

Sensor Types
^^^^^^^^^^^^

.. doxygengroup:: ovrtx_sensor_types
   :project: ovrtx
   :content-only:

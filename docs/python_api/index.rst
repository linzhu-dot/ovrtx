.. SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
.. SPDX-License-Identifier: LicenseRef-NvidiaProprietary
..
.. NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
.. property and proprietary rights in and to this material, related
.. documentation and any modifications thereto. Any use, reproduction,
.. disclosure or distribution of this material and related documentation
.. without an express license agreement from NVIDIA CORPORATION or
.. its affiliates is strictly prohibited.

Python API Reference
====================

The Python API provides a high-level interface to the ovrtx rendering library.

Known Limitations
-----------------

In the current version, you **cannot** use the ``usd-core`` package from PyPI. If you wish to use OpenUSD in the same process as ovrtx, you must use the same version of OpenUSD as the one used to build ovrtx. This means that you must use `OpenUSD v25.11 <https://github.com/PixarAnimationStudios/OpenUSD/tree/v25.11>`_, built as non-monolithic libraries with Python support enabled (the default). You must also use oneTBB v2021.13.0 or later when building OpenUSD.

This limitation will be lifted in a future version of ovrtx.


.. contents:: Contents
   :local:
   :depth: 2

ovrtx
-----------

.. automodule:: ovrtx
   :members:
   :undoc-members:
   :show-inheritance:
   :inherited-members:

ovrtx.math
----------

.. automodule:: ovrtx.math
   :members:
   :undoc-members:
   :show-inheritance:

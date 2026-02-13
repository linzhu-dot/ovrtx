.. SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
.. SPDX-License-Identifier: LicenseRef-NvidiaProprietary
..
.. NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
.. property and proprietary rights in and to this material, related
.. documentation and any modifications thereto. Any use, reproduction,
.. disclosure or distribution of this material and related documentation
.. without an express license agreement from NVIDIA CORPORATION or
.. its affiliates is strictly prohibited.

Examples
========

This section contains example projects demonstrating various features of ovrtx.

C Examples
----------

.. list-table::
   :widths: 50 50
   :header-rows: 0

   * - .. image:: ../../img/example-minimal.jpg
          :alt: Minimal Example
          :target: c_minimal.html

       **Minimal**

       Basic renderer initialization, rendering a single frame from an RGB camera and writing the result to disk as a PNG.

       :doc:`Build & run instructions → <c_minimal>`

     - .. image:: ../../img/example-vulkan-interop.gif
          :alt: Vulkan Interop Example
          :target: c_vulkan_interop.html

       **Vulkan Interop**

       Demonstrates ovrtx-Vulkan interoperability, rendering USD scenes and displaying them in a Vulkan window with interactive orbit camera control.

       :doc:`Build & run instructions → <c_vulkan_interop>`

Python Examples
---------------

.. list-table::
   :widths: 50 50
   :header-rows: 0

   * - .. image:: ../../img/example-minimal.jpg
          :alt: Minimal Example
          :target: python_minimal.html

       **Minimal**

       Basic workflow: create a Renderer, load a USD layer, step the renderer, and map/display the rendered output.

       :doc:`Build & run instructions → <python_minimal>`

     - .. image:: ../../img/example-planet-system.jpg
          :alt: Planet System Example
          :target: python_planet_system.html

       **Planet System**

       Animated planetary system using Warp kernels for GPU-accelerated animation, demonstrating dynamic scene modification and zero-copy transform updates.

       :doc:`Build & run instructions → <python_planet_system>`

.. toctree::
   :hidden:

   c_minimal
   c_vulkan_interop
   python_minimal
   python_planet_system

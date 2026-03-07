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

Python Examples
---------------

.. grid:: 2
   :gutter: 2

   .. grid-item-card:: Minimal
      :img-top: ../../img/example-minimal.jpg
      :img-alt: Minimal example output

      Basic workflow: create a Renderer, load a USD layer, step the renderer, and map/display the rendered output.
      +++
      :doc:`Build & run instructions → <python_minimal>`

   .. grid-item-card:: Planet System
      :img-top: ../../img/example-planet-system.jpg
      :img-alt: Planet System example output

      Animated planetary system using Warp kernels for GPU-accelerated animation, demonstrating dynamic scene modification and zero-copy transform updates.
      +++
      :doc:`Build & run instructions → <python_planet_system>`

C Examples
----------

.. grid:: 2
   :gutter: 2

   .. grid-item-card:: Minimal
      :img-top: ../../img/example-minimal.jpg
      :img-alt: Minimal example output

      Basic renderer initialization, rendering a single frame from an RGB camera and writing the result to disk as a PNG.
      +++
      :doc:`Build & run instructions → <c_minimal>`

   .. grid-item-card:: Vulkan Interop
      :img-top: ../../img/example-vulkan-interop.gif
      :img-alt: Vulkan Interop example output

      Demonstrates ovrtx-Vulkan interoperability, rendering USD scenes and displaying them in a Vulkan window with interactive orbit camera control.
      +++
      :doc:`Build & run instructions → <c_vulkan_interop>`

.. toctree::
   :hidden:

   python_minimal
   python_planet_system
   c_minimal
   c_vulkan_interop

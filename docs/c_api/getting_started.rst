.. SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
.. SPDX-License-Identifier: LicenseRef-NvidiaProprietary
..
.. NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
.. property and proprietary rights in and to this material, related
.. documentation and any modifications thereto. Any use, reproduction,
.. disclosure or distribution of this material and related documentation
.. without an express license agreement from NVIDIA CORPORATION or
.. its affiliates is strictly prohibited.

Getting Started in C
====================

The C/C++ examples require CMake and a development environment. On Windows this is provided by `Visual Studio 2017 or newer <https://visualstudio.microsoft.com/>`_. On Linux (Ubuntu):

.. code-block:: bash

   sudo apt-get install build-essential cmake

To get started, first clone `the repository <https://github.com/NVIDIA-Omniverse/ovrtx>`__ and build and run the first example using CMake:

.. code-block:: bash

   git clone https://github.com/NVIDIA-Omniverse/ovrtx.git
   cd ovrtx/examples/c/minimal
   cmake -B build

Then, on Windows:

.. code-block:: text

   cmake --build build --config Release
   .\build\Release\minimal.exe

On Linux:

.. code-block:: bash

   cmake --build build --config Release
   ./build/minimal

The minimal example shows how to create the renderer, load an OpenUSD scene and render a single image, copying the results back to the CPU for writing out as a PNG.

.. image:: ../../img/example-minimal.jpg
   :alt: Minimal example output
   :align: center

The resulting image will be written to ``./out.png`` and can be inspected with any image viewer.

Note that the first time a program built against ovrtx is run, it will compile and cache necessary shaders, which may take some time depending on your system. Subsequent runs will use the cached shaders and will be fast.

Installation
------------

CMake
^^^^^

ovrtx binary distributions can be found on the GitHub `Releases page <https://github.com/NVIDIA-Omniverse/ovrtx/releases>`__, and contain a CMake config.

Alternatively, download the appropriate package for your system from the `Releases page <https://github.com/NVIDIA-Omniverse/ovrtx/releases>`__ and point
``CMAKE_PREFIX_PATH`` to the directory where you extracted the archive and use ``find_package(ovrtx)``
from your ``CMakeLists.txt``.

For other build systems, download the appropriate package for your system from the `Releases page <https://github.com/NVIDIA-Omniverse/ovrtx/releases>`__. The headers are in the
``include`` directory and libraries are in ``lib`` and ``bin``, in either static or dynamic flavors.

The simplest way to add ovrtx as a dependency to your project is using CMake FetchContent:

.. literalinclude:: ../../examples/c/cmake/ovrtx.cmake
   :language: cmake
   :lines: 27-83

Note that the macro above is provided for convenience in ``ovrtx.cmake`` in the ``examples/c/cmake`` directory in `the repository <https://github.com/NVIDIA-Omniverse/ovrtx>`__.

Runtime Packaging and Deployment
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

ovrtx requires several libraries and other runtime dependencies to be present and discoverable at runtime. These are all included in the ovrtx binary distribution under the ``bin`` directory:

.. code-block:: text

   bin/
   ├── libovrtx-dynamic.so / ovrtx-dynamic.dll
   ├── cache/
   ├── library/
   ├── libs/
   ├── mdl/
   ├── plugins/
   ├── rendering-data/
   └── usd_plugins/

The ovrtx dynamic library will automatically load the other dependencies at runtime if it is placed alongside them as in the binary distribution. If you need to deploy your application with a different layout, you can point ovrtx to the correct paths using the :c:func:`ovrtx_config_entry_binary_package_root_path` helper function when configuring the renderer:

.. code-block:: c

   ovrtx_renderer_config_entry_t config_entries[] = {
       ovrtx_config_entry_binary_package_root_path(ovx_string("/path/where/bin/contents/live"))
   };

   ovrtx_config_t config;
   config.entries = config_entries;
   config.entry_count = sizeof(config_entries) / sizeof(config_entries[0]);

   ovrtx_renderer_t* renderer;
   ovrtx_result_t result = ovrtx_create_renderer(&config, &renderer);

Note that when static linking ovrtx, you MUST provide the binary package root path or ovrtx will not be able to find the required dependencies at runtime.

Minimal Example
---------------

.. literalinclude:: ../../examples/c/minimal/main.cpp
   :language: cpp
   :lines: 11-

.. image:: ../../img/example-minimal.jpg
   :alt: Minimal example output
   :align: center

Next Steps
----------

* Explore more :doc:`../examples/index` including the :doc:`Vulkan Interop <../examples/c_vulkan_interop>` example with real-time GPU rendering.
* See the :doc:`index` for the full C API reference.

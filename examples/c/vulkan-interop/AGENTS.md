# AGENTS.md - AI Agent Guidelines for ovrtx-interop

> **IMPORTANT**: AI agents must update this file after making significant changes to the project structure, architecture, or conventions. This ensures future agents have accurate context.

## Project Overview

This is a CUDA-Vulkan interop application demonstrating:
- Sharing GPU memory between CUDA and Vulkan
- Using ovrtx library for USD scene rendering via CUDA
- Rendering ovrtx output as a Vulkan texture
- **Double-buffered async rendering** with timeline semaphores
- Interactive orbit camera control with mouse drag
- Build-time GLSL shader compilation to SPIR-V via glslc
- Bindless texture access using descriptor indexing

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                        Initialization                        │
├─────────────────────────────────────────────────────────────┤
│  1. ovrtx init → load USD scene (creates CUDA context)      │
│  2. CUDA Driver API init → get device UUID from ovrtx ctx   │
│  3. Load precompiled SPIR-V shaders from build/shaders/     │
│  4. Vulkan init → match physical device by UUID             │
│  5. Create 2 exportable Vulkan images (double-buffer)       │
│  6. Create timeline semaphore for CUDA-Vulkan sync          │
│  7. Export memory FDs → import into CUDA as surfaces        │
│  8. Create OrbitCamera for interactive control              │
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│                  Async Render Loop (Ping-Pong)              │
├─────────────────────────────────────────────────────────────┤
│  1. Poll mouse input → update OrbitCamera                   │
│  2. Check if CUDA finished previous frame (cuEventQuery)    │
│  3. If done → swap read_idx/write_idx buffers               │
│  4. Start CUDA: ovrtx_step() → copy to image[write_idx]     │
│  5. Signal timeline semaphore after CUDA copy               │
│  6. Vulkan reads from image[read_idx] (no wait!)            │
│  7. Vulkan renders fullscreen quad → present                │
│  (CUDA and Vulkan run in parallel)                          │
└─────────────────────────────────────────────────────────────┘
```

## Double-Buffering with Timeline Semaphores

The render loop uses async double-buffering to prevent CPU stalls:

- **Two shared images**: CUDA writes to one while Vulkan reads the other
- **Timeline semaphore**: CUDA signals after each frame; CPU polls to detect completion
- **No blocking syncs**: `cuStreamSynchronize` is not called in the hot path
- **Ping-pong swap**: When CUDA finishes, buffers swap so Vulkan gets fresh data

```
Frame N:   CUDA writes image[0], signals timeline=1
           Vulkan reads image[1] (previous data)
           
Frame N+1: CPU polls timeline, sees >=1 → swap indices
           CUDA writes image[1], signals timeline=2
           Vulkan reads image[0] (freshly written)
```

## File Structure

```
ovrtx-interop/
├── CMakeLists.txt           # Build config, FetchContent for dependencies
├── justfile                 # Just command runner tasks
├── AGENTS.md                # This file - AI agent guidelines
├── README.md                # User-facing documentation
├── src/
│   ├── main.cpp             # Application entry, CLI, render loop, ovrtx integration
│   ├── vk/
│   │   ├── vulkan_context.cpp/hpp  # VulkanContext class
│   │   ├── sampled_image.cpp/hpp   # SampledImage resource management
│   │   ├── shader.cpp/hpp          # Shader object management
│   │   └── command_buffer.cpp/hpp  # CommandBuffer wrapper
│   ├── cuda/
│   │   └── cuda_kernel.cpp/hpp     # CUDA Driver API + memory copy
│   ├── glsl/
│   │   └── spirv_loader.hpp        # Load precompiled SPIR-V files
│   └── camera/
│       └── orbit_camera.cpp/hpp    # OrbitCamera for mouse control
├── ovrtx/                   # ovrtx library (bin/, include/, lib/)
├── shaders/
│   ├── fullscreen.vert      # Vertex shader (GLSL)
│   └── fullscreen.frag      # Fragment shader (GLSL, bindless textures)
├── torus-plane.usda         # Example USD scene for testing
└── tests/
    ├── validation_tracker.cpp/hpp  # Vulkan validation layer tracking
    ├── test_context.cpp     # VulkanContext creation tests
    ├── test_descriptors.cpp # Descriptor indexing tests
    ├── test_rendering.cpp   # Rendering and swapchain tests
    ├── test_cuda_kernel.cpp # CUDA kernel tests
    └── test_cuda_interop.cpp # CUDA-Vulkan interop tests
```

## Coding Conventions

- **Naming**: snake_case for variables and functions, PascalCase for types
- **Language**: C++17 for all code (no .cu files, uses Driver API)
- **Error handling**: VK_CHECK() macro for Vulkan, CU_CHECK() macros for CUDA Driver API
- **Globals**: Prefixed with `g_` (e.g., `g_cuda_external_memory`)
- **Constants**: UPPER_SNAKE_CASE (e.g., `TEX_WIDTH`, `WINDOW_HEIGHT`)
- **East const**: Use `int const*` not `const int*`
- **Auto usage**: Only use `auto` for trailing return type syntax (`auto foo() -> ReturnType`) and structured bindings (`auto [a, b] = ...`), never for other variable declarations

## CUDA Driver API

This project uses the CUDA Driver API (not Runtime API):
- `cuInit`, `cuDeviceGet`, `cuCtxCreate` for initialization
- `cuImportExternalMemory` for Vulkan memory import
- `cuSurfObjectCreate` for surface objects
- `cuLaunchKernel` for kernel execution

## NVRTC Runtime Compilation

The CUDA kernel is compiled at runtime:
- Kernel source stored as string constant in `cuda_kernel.cpp`
- `nvrtcCreateProgram` and `nvrtcCompileProgram` generate PTX
- `cuModuleLoadData` loads PTX into driver
- `cuModuleGetFunction` retrieves kernel function pointer

## Shader Compilation

GLSL shaders are compiled to SPIR-V at build time:
- CMake uses `glslc` from the Vulkan SDK
- Source shaders in `shaders/` directory
- Compiled `.spv` files output to `build/shaders/` and copied next to the executable
- `spirv_loader.hpp` provides `load_spirv()` function to read compiled shaders
- `get_executable_dir()` resolves shader path relative to the binary (works on Linux and Windows)

## Vulkan 1.3 Requirements

This project requires Vulkan 1.3 for dynamic rendering:
- Uses `VK_API_VERSION_1_3`
- Enables `VkPhysicalDeviceDynamicRenderingFeatures`
- No VkRenderPass or VkFramebuffer objects
- Uses `vkCmdBeginRendering()` / `vkCmdEndRendering()`

## VK_EXT_shader_object

This project uses shader objects instead of pipelines:
- Enables `VkPhysicalDeviceShaderObjectFeaturesEXT`
- Uses `VkShaderEXT` instead of `VkPipeline`
- Shaders created with `vkCreateShadersEXT` from SPIR-V
- Shaders bound with `vkCmdBindShadersEXT`
- All state is dynamic (viewport, scissor, rasterization, blend, etc.)
- Extension functions loaded via `vkGetDeviceProcAddr`

Dynamic state commands used:
- `vkCmdSetViewportWithCount`, `vkCmdSetScissorWithCount`
- `vkCmdSetRasterizerDiscardEnable`, `vkCmdSetPolygonModeEXT`
- `vkCmdSetCullMode`, `vkCmdSetFrontFace`, `vkCmdSetDepthBiasEnable`
- `vkCmdSetPrimitiveTopology`, `vkCmdSetPrimitiveRestartEnable`
- `vkCmdSetDepthTestEnable`, `vkCmdSetDepthWriteEnable`
- `vkCmdSetRasterizationSamplesEXT`, `vkCmdSetSampleMaskEXT`
- `vkCmdSetColorBlendEnableEXT`, `vkCmdSetColorWriteMaskEXT`
- `vkCmdSetVertexInputEXT`

## Key Vulkan Extensions

Instance extensions:
- VK_KHR_external_memory_capabilities
- VK_KHR_external_semaphore_capabilities
- VK_KHR_get_physical_device_properties_2

Device extensions:
- VK_KHR_swapchain
- VK_KHR_external_memory
- VK_KHR_external_memory_fd
- VK_KHR_external_semaphore
- VK_KHR_external_semaphore_fd
- VK_EXT_shader_object

## CUDA Interop Types (Driver API)

- `CUexternalMemory` - Handle to imported Vulkan memory (array of 2 for double-buffer)
- `CUmipmappedArray` - Mapped from external memory (array of 2)
- `CUsurfObject` - Used for kernel writes via surf2Dwrite()
- `CUexternalSemaphore` - Timeline semaphore for async CUDA-Vulkan sync
- `CUevent` - Used to poll CUDA frame completion without blocking

## Build Commands

```bash
# Using just (recommended)
just configure   # cmake -B build -G Ninja -S .
just build       # cmake --build build
just test        # ctest --test-dir build --output-on-failure

# Manual cmake commands
cmake -B build -G Ninja
cmake --build build

# Run application
./build/ovrtx-interop <usd_file_path> <render_product_path>

# Example with included scene
./build/ovrtx-interop torus-plane.usda /Render/Camera

# Run tests
ctest --test-dir build --output-on-failure
```

Note: First build will download GoogleTest, GLM, volk, and unordered_dense via FetchContent.

## Dependencies

**System packages** (must be in CMAKE_PREFIX_PATH):
- Vulkan SDK 1.4+ (provides glslc compiler)
- CUDA Toolkit
- glfw3
- ovrtx library

**FetchContent downloads** (automatic):
- GoogleTest v1.14.0
- GLM v1.0.3
- volk v1.4.304
- unordered_dense v4.4.0

## Command Line Interface

```
Usage: ovrtx-interop [options] <usd_file_path> <render_product_path>

Arguments:
  usd_file_path       Path to the USD scene file
  render_product_path USD path to the render product (camera/sensor)

Options:
  --up Y|Z       Up axis for scene coordinate system (default: Y)
  --units m|cm   Scene units for camera distance scaling (default: cm)
```

### Camera Distance Scaling

The orbit camera's initial distance is scaled based on the `--units` option:
- **Centimeters (default)**: Distance = ~1732 cm (appropriate for cm-scale scenes)
- **Meters**: Distance = ~17.3 m (1/100th of cm distance)

### Up Axis Support

The `--up` option configures the orbit camera's vertical axis:
- **Y-up (default)**: Standard for many DCC tools (Maya, Blender default)
- **Z-up**: Common in CAD/engineering applications (AutoCAD, SolidWorks)

## ovrtx Integration

The ovrtx library provides USD scene rendering via CUDA. Integration is handled directly in `main.cpp`:
- Uses ovrtx C API directly (no wrapper class)
- Supports two output types (prefers HdrColor, falls back to LdrColor):
  - **HdrColor**: RGBA 16-bit float (half-precision), linear color space
  - **LdrColor**: RGBA 8-bit uint, sRGB color space
- Output type is detected at startup and determines Vulkan/CUDA formats
- Output is provided as CUarray via `OVRTX_MAP_DEVICE_TYPE_CUDA_ARRAY`
- `cuda_copy_array_to_surface(buffer_idx, ..., format, ...)` copies to double-buffered image
- Error handling via `ovrtx_get_last_error()` function
- Transform updated via `ovrtx_write_attribute()` with `OVRTX_SEMANTIC_TRANSFORM_4x4`

### Output Format Mapping

| Output Type | Vulkan Format | CUDA Format | Bytes/Pixel |
|-------------|--------------|-------------|-------------|
| HdrColor | `VK_FORMAT_R16G16B16A16_SFLOAT` | `CU_AD_FORMAT_HALF` | 8 |
| LdrColor | `VK_FORMAT_R8G8B8A8_SRGB` | `CU_AD_FORMAT_UNSIGNED_INT8` | 4 |

The `CudaImageFormat` enum in `cuda_kernel.hpp` encapsulates this:
- `CudaImageFormat::Half4` - For HdrColor (16-bit float × 4 channels)
- `CudaImageFormat::UInt8_4` - For LdrColor (8-bit uint × 4 channels)

Data flow (async double-buffered):
```
OrbitCamera::update() → ovrtx_write_attribute(transform)
      ↓
ovrtx_step() → ovrtx_fetch_results() → ovrtx_map_rendered_output()
      ↓
cuStreamWaitEvent(ovrtx.wait_event)
      ↓
cuda_copy_array_to_surface(write_idx, CUarray)
      ↓
cuda_signal_timeline(frame_counter)  ← async signal
      ↓
ovrtx_unmap_rendered_output()
      ↓
[Meanwhile] Vulkan renders image[read_idx]
```

## Orbit Camera Control

Interactive camera control via mouse:
- Left-click-and-drag rotates camera around target point
- Scroll wheel dollies camera in/out (adjusts distance to target)
- Uses spherical coordinates (azimuth, elevation, distance)
- `OrbitCamera` class in `src/camera/orbit_camera.hpp`
- GLFW callbacks track mouse state
- Camera transform sent to ovrtx via `ovrtx_write_attribute()`

## Multi-GPU Notes

This project is designed for systems with multiple GPUs:
- CUDA is initialized first to establish the target device
- Vulkan device selection matches the CUDA device UUID
- UUID comparison uses VkPhysicalDeviceIDProperties

## Platform Support

Currently Linux-only. Windows support planned.

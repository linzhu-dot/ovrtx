# vulkan-interop

Demonstrates how to integrate ovrtx with Vulkan by sharing renders on the GPU. 

ovrtx outputs are mapped to CUDA arrays every frame, which are then copied to CUDA-exported VkImage memory. The resulting textures are then sampled on a fullscreen quad to display the render in real time in a GLFW window. Memory access between CUDA and Vulkan is synchronized using timeline semaphores.

![example-vulkan-interop](../../../img/example-vulkan-interop.gif)

## Linux

### Prerequisites

- `sudo apt install build-essential cmake`
- [Vulkan SDK 1.3.250+](https://vulkan.lunarg.com/sdk/home)
- [CUDA Toolkit 12.0+](https://developer.nvidia.com/cuda-downloads)

Other dependencies (ovrtx, GLM, volk, unordered_dense, glfw3) are downloaded automatically at configure time via CMake FetchContent.

### Building

```bash
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build
```

### Running

```bash
./build/ovrtx-interop
```

## Windows

### Prerequisites

- [Visual Studio 2017+](https://visualstudio.microsoft.com/downloads/)
- [Vulkan SDK 1.3.250+](https://vulkan.lunarg.com/sdk/home)
- [CUDA Toolkit 12.0+](https://developer.nvidia.com/cuda-downloads)

Other dependencies (ovrtx, GLM, volk, unordered_dense, glfw3) are downloaded automatically at configure time via CMake FetchContent.

### Building

```pwsh
cmake -B build
cmake --build --config Release
```

### Running

```pwsh
.\build\Release\ovrtx-interop.exe
```

The example is configured to load the robot scene from Omniverse:

| Setting | Value |
|---------|-------|
| USD Scene | `https://omniverse-content-production.s3.us-west-2.amazonaws.com/Samples/Robot-OVRTX/robot-ovrtx.usda` |
| Render Product | `/Render/Camera` |
| Up Axis | Z |
| Units | Meters |

### Controls

- **Left-click and drag** — Rotate camera around the target point
- **Mouse wheel** — Dolly camera in/out

# Licensing

This example contains stb_image_write.h, © Sean Barrett, released under Public Domain.

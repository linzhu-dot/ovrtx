# Minimal Example

This example shows basic initialization of the renderer, rendering a single frame from an RGB camera, mapping the output and writing the result to disk as a PNG.

The example will load an example scene from S3 and output the resulting image to `out.png`.

The first time the example is run, driver shader compilation will be performed and cached. Subsequent runs will be much faster. 

![output](../../../img/out.jpg)


## Linux

### Prerequisites

- `sudo apt install build-essential cmake`

### Building

```bash
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build
```

### Running

```bash
./build/minimal
```

## Windows

### Prerequisites

- [Visual Studio 2017+](https://visualstudio.microsoft.com/downloads/)

### Building

```pwsh
cmake -B build
cmake --build --config Release
```

### Running

```pwsh
.\build\Release\minimal
```

# Licensing

This example contains stb_image_write.h, © Sean Barrett, released under Public Domain. 
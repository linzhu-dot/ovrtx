# Minimal ovrtx Example

This is the minimal Python example from the ovrtx README. It demonstrates the basic workflow:

1. Create a Renderer
2. Load a USD layer (with an inline USDA that references a remote scene)
3. Step the renderer to produce a frame
4. Map the rendered output and display it

![output](../../../img/example-minimal.jpg)

## Prerequisites

- Python 3.10–3.13
- [uv](https://docs.astral.sh/uv/)


## Running

```bash
uv run main.py
```

The first run will take some time as shaders are compiled and cached.

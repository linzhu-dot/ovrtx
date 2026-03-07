# ovrtx Python Tests

pytest-based tests for the ovrtx Python bindings.

## Prerequisites

- Python 3.10–3.13
- NVIDIA GPU with RTX support
- NVIDIA driver supporting CUDA 12+
- ovrtx package (C library + Python bindings)

## Environment Setup

The tests require the ovrtx C library and Python bindings to be accessible. Set up your environment based on your installation method:

### From Package Distribution

If you have the packaged ovrtx distribution (e.g., `ovrtx.release/`):

**Linux:**
```bash
export LD_LIBRARY_PATH=/path/to/ovrtx.release/bin:$LD_LIBRARY_PATH
export PYTHONPATH=/path/to/ovrtx.release/python:$PYTHONPATH
```

**Windows (PowerShell):**
```powershell
$env:PATH = "C:\path\to\ovrtx.release\bin;$env:PATH"
$env:PYTHONPATH = "C:\path\to\ovrtx.release\python;$env:PYTHONPATH"
```

**Windows (Command Prompt):**
```cmd
set PATH=C:\path\to\ovrtx.release\bin;%PATH%
set PYTHONPATH=C:\path\to\ovrtx.release\python;%PYTHONPATH%
```

### From pip/wheel Installation

If ovrtx is installed via pip, only the library path is needed:

**Linux:**
```bash
export LD_LIBRARY_PATH=/path/to/ovrtx/bin:$LD_LIBRARY_PATH
```

**Windows:**
```powershell
$env:PATH = "C:\path\to\ovrtx\bin;$env:PATH"
```

## Environment Variables

| Variable | Required | Description |
|----------|----------|-------------|
| `LD_LIBRARY_PATH` (Linux) | Yes | Must include path to `libovrtx-dynamic.so` |
| `PATH` (Windows) | Yes | Must include path to `ovrtx-dynamic.dll` |
| `PYTHONPATH` | Conditional | Required if ovrtx not installed via pip |

## Installing Dependencies

```bash
# Create and activate virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # Linux/macOS
# or: venv\Scripts\activate  # Windows

# Install test dependencies
pip install -r requirements.txt
```

## Running Tests

### Via pytest (Recommended)

Run pytest from the directory containing the `ovrtx` package (e.g., `ovrtx.release/python/`):

```bash
# Run all tests (uses fallback path for test data)
pytest ovrtx/tests/

# Specify test data directory explicitly
pytest ovrtx/tests/ --test-data=/path/to/usd/test/scenes

# Run with verbose output
pytest ovrtx/tests/ -v

# Run specific test
pytest ovrtx/tests/ -k test_renderer

# Run tests matching pattern
pytest ovrtx/tests/ -k "map_attribute"

# Generate JUnit XML report
pytest ovrtx/tests/ --junit-xml=results.xml

# Custom output directory for test artifacts
pytest ovrtx/tests/ --output=/path/to/output
```

## Command-Line Options

| Option | Default | Description |
|--------|---------|-------------|
| `--test-data DIR` | (fallback) | Directory containing USD test scenes (e.g., `cube.usda`) |
| `--output DIR` | `tests/_output` | Directory for test output files (debug dumps, images) |
| `-k EXPR` | — | Run tests matching expression (pytest built-in) |
| `-v` / `-vv` | — | Verbose / very verbose output (pytest built-in) |
| `--junit-xml FILE` | — | Generate JUnit XML report (pytest built-in) |

## Test Structure

```
ovrtx/tests/
├── README.md           # This file
├── requirements.txt    # Python dependencies (pytest, numpy, warp-lang, etc.)
├── conftest.py         # pytest fixtures and configuration
├── test_ovrtx.py       # Main test file with all test functions
└── _output/            # Test output directory (auto-created)
    └── test_ovrtx.log  # OVRTX log file (created at runtime)
```

## Log File Output

Tests generate a log file at `<output_dir>/test_ovrtx.log` (default: `ovrtx/tests/_output/test_ovrtx.log`). Crash dumps go to the same directory as the log file. Use `--output` to change the output directory.

## Troubleshooting

### Library Not Found

```
OSError: libovrtx-dynamic.so: cannot open shared object file
```

**Solution:** Ensure `LD_LIBRARY_PATH` (Linux) or `PATH` (Windows) includes the directory containing the ovrtx C library.

### Test Data Not Found

```
pytest.fail: USD scene file not found: .../cube.usda
```

**Solution:** Use `--test-data` option to specify the directory containing USD test scenes:
```bash
pytest ovrtx/tests/ --test-data=/path/to/usd/test/scenes
```

### Import Error

```
ModuleNotFoundError: No module named 'ovrtx'
```

**Solution:** Ensure `PYTHONPATH` includes the parent directory of the `ovrtx` package, or install ovrtx via pip.

### CUDA Errors

```
RuntimeError: CUDA error: no CUDA-capable device is detected
```

**Solution:** Verify NVIDIA driver is installed and GPU is RTX-capable. Check with `nvidia-smi`.

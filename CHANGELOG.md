# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/) and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.2.0] - 2026-03-06

### Added

- Support for Gaussian Splats and other particle field primitive types using the [`UsdVol.ParticleField`](https://openusd.org/dev/user_guides/schemas/usdVol/ParticleField.html) schema.
- Operation status query API and logging callback for monitoring renderer operations.
- Dedicated functions for creating supported config values.
- GPU selection by CUDA device index at renderer creation using `ovrtx_config_entry_active_cuda_gpus()` and per render product using `uint[] deviceIds`.
- `ovrtx_get_version()` Python binding with version compatibility check between the Python package and native library.
- Python bindings for `enable_profiling`, `read_gpu_transforms` config entries.
- Async data access in Python bindings (`write_attribute`, `write_array_attribute`) matching the C API.

### Changed

- Upgraded DLPack from 0.8 to 1.3 in both C and Python APIs, allowing creating boolean tensors.
- Attribute writes now accept any memory-compatible tensors as input. This means that a N-element, 4x4 matrix can now be
  written as shape=[N, 4, 4] instead of shape[N] with 16 lanes as was previously required, making interop with numpy
  much simpler and removing the need for extra helper functions to correct tensor shapes.
- C API headers are now pure C compatible (removed C++ constructs from `ovrtx_attributes.h` and `pathdictionary_helper.h`)
- Transform attributes now use the `omni:xform` alias; direct writing of `localMatrix`/`worldMatrix` is no longer
  supported and transforms must be written to `omni:xform` instead. If those transforms are in world space as opposed to
  local space then `bool resetXformStack=true` must also be set on the same prim.

### Fixed

- Removed spurious `IRenderSettings::getRenderSettings failed` warning when no global RenderSettings prim is present in the USD stage.
- Removed nvidia-smi printout on startup.
- Fixed material-related memory leaks.
- Fixed a visual glitch when calling `ovrtx_reset_stage()`.

### Security

- Updated OpenSSL to address CVE-2025-15467


## [0.1.0] - 2026-02-13

### Added
- Initial release of ovrtx: NVIDIA Omniverse RTX Rendering library
- C library
- Python bindings
- Example source code
- Documentation

### Limitations

- ovrtx cannot be used in a process that also links OpenUSD that is not v25.11, non-monolithic and Python-enabled, linked against oneTBB. In particular, this means ovrtx cannot be used together with `usd-core` in the same process. This limitation will be lifted in a future version.
- ovrtx curently supports camera sensors only. Other types of sensors will be supported in the next minor release.

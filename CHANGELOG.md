# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/) and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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

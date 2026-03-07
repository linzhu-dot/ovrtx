# Copyright (c) 2025-2026, NVIDIA CORPORATION. All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

"""Pytest configuration for ovrtx tests."""

from pathlib import Path

import pytest


def pytest_addoption(parser):
    """Register custom pytest command-line options."""
    parser.addoption(
        "--output",
        action="store",
        default=str(Path(__file__).parent / "_output"),
        help="Directory for test output files (defaults to tests/_output)",
    )
    parser.addoption(
        "--test-data",
        action="store",
        default=str(Path(__file__).parent / "data"),
        help="Directory containing USD test data files (defaults to data/usd/tests/ovrtx in repo)",
    )


def pytest_configure(config):
    """Configure pytest with custom markers."""
    config.addinivalue_line("markers", "usd_scene(name): specify USD scene file to load for the test")


@pytest.fixture(scope="session", autouse=True)
def test_output_dir(request) -> Path:
    """
    Fixture that provides the test output directory.

    Can be overridden via --output command-line option, otherwise uses default.
    This fixture is autouse, so all tests can access it without explicit declaration.

    Returns:
        Path to test output directory (created if doesn't exist)
    """
    output_path = request.config.getoption("--output")

    if output_path:
        output_dir = Path(output_path).resolve()
    else:
        # Fallback: _output directory next to test files
        output_dir = Path(__file__).parent / "_output"

    # Ensure directory exists
    output_dir.mkdir(parents=True, exist_ok=True)

    return output_dir


@pytest.fixture
def usd_scene(request) -> Path:
    """
    Fixture that provides the resolved path to a USD test scene.

    Usage:
        @pytest.mark.usd_scene("cube.usda")
        def test_something(usd_scene):
            renderer.add_usd(str(usd_scene))

    Returns:
        Path to the USD scene file (resolved and validated to exist)

    Raises:
        pytest.fail if scene name not specified via marker or scene file doesn't exist
    """
    marker = request.node.get_closest_marker("usd_scene")
    if marker is None:
        pytest.fail("Test must be decorated with @pytest.mark.usd_scene('scene_name.usda')")

    # Resolve test data directory: CLI option > fallback
    data_path = Path(request.config.getoption("--test-data"))
    scene_name = marker.args[0]
    scene_path = (data_path / scene_name).resolve()

    if not scene_path.exists():
        pytest.fail(f"USD scene file not found: {scene_path}")

    return scene_path


@pytest.fixture(scope="session")
def shared_renderer(test_output_dir):
    """Session-scoped renderer singleton.

    Creates one renderer for the entire test session to avoid exhausting
    GPU SyncScope IDs (see Appendix A in plan_stage_management_apis.md).

    Tests should use the `renderer` fixture which calls reset_stage() for cleanup.
    """
    from ovrtx import Renderer, RendererConfig

    # Update log file path to use test_output_dir from CLI option
    # [snippet:renderer-config]
    _OVRTX_TEST_CONFIG = RendererConfig(
        # Configure logging via OVRTX config
        log_level="info",
        log_file_path=str(test_output_dir / "test_ovrtx.log"),
    )

    renderer = Renderer(_OVRTX_TEST_CONFIG)
    # [/snippet:renderer-config]
    yield renderer
    del renderer


@pytest.fixture
def renderer(shared_renderer):
    """Per-test renderer with clean stage.

    Resets the shared renderer's stage before each test to ensure isolation.
    Tests load their own USD scenes via renderer.add_usd(str(usd_scene)).
    """
    shared_renderer.reset_stage()
    shared_renderer.reset(time=0.0)
    assert shared_renderer, "Renderer should be valid"
    return shared_renderer

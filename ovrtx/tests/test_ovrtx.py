# Copyright (c) 2025-2026, NVIDIA CORPORATION. All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

"""
pytest tests for ovrtx Python bindings.

These tests can be run standalone:
    python -m venv venv
    source venv/bin/activate  # or venv\\Scripts\\activate on Windows
    pip install -r requirements.txt
    pytest
"""

from pathlib import Path

import pytest
import warp as wp

from ovrtx import Renderer, RendererConfig

wp.init()

TEST_OUTPUT_DIR = Path(__file__).parent / "_output"

# Default startup options for ovrtx tests
# Note: Paths are updated in shared_renderer fixture (conftest.py) using test_output_dir
OVRTX_TEST_CONFIG = RendererConfig(
    # Configure logging via OVRTX config
    log_file_path=str(TEST_OUTPUT_DIR / "test_ovrtx.log"),
    log_level="info",
)


@wp.func
def get_pixel_value(img: wp.array2d(dtype=wp.vec4ub), i: int, j: int) -> float:
    i = wp.clamp(i, 0, img.shape[0] - 1)
    j = wp.clamp(j, 0, img.shape[1] - 1)
    p = img[i, j]
    return (float(p[0]) + float(p[1]) + float(p[2])) / 3.0


@wp.kernel
def sobel_kernel(img: wp.array2d(dtype=wp.vec4ub), output: wp.array2d(dtype=wp.uint8)):
    i, j = wp.tid()

    p00 = get_pixel_value(img, i - 1, j - 1)
    p01 = get_pixel_value(img, i - 1, j)
    p02 = get_pixel_value(img, i - 1, j + 1)
    p10 = get_pixel_value(img, i, j - 1)
    # p11 = get_pixel_value(img, i, j)
    p12 = get_pixel_value(img, i, j + 1)
    p20 = get_pixel_value(img, i + 1, j - 1)
    p21 = get_pixel_value(img, i + 1, j)
    p22 = get_pixel_value(img, i + 1, j + 1)

    ex = p02 + (2.0 * p12) + p22 - (p00 + (2.0 * p10) + p20)
    ey = p00 + (2.0 * p01) + p02 - (p20 + (2.0 * p21) + p22)
    e = wp.sqrt(ex * ex + ey * ey)

    output[i, j] = wp.uint8(wp.clamp(e, 0.0, 255.0))


def get_stage_debug_dump(renderer: Renderer) -> str:
    """Get USDA stage dump for verification.

    Runs a step (with delta_time=0) to capture the ovrtx_debug_dump_stage product and returns the decoded string.
    """
    products = renderer.step(render_products={"ovrtx_debug_dump_stage"}, delta_time=0.0)
    assert products is not None, "Debug dump products should be valid"
    assert "ovrtx_debug_dump_stage" in products, "ovrtx_debug_dump_stage product should exist"
    frame = products["ovrtx_debug_dump_stage"].frames[0]
    assert "debug" in frame.render_vars, "debug render var should exist"
    with frame.render_vars["debug"].map(device="cpu") as mapping:
        return mapping.tensor.to_bytes().decode("utf-8")


def test_ovx_string_t():
    from ovrtx._src.bindings import ovx_string_t

    print(f"ovx_string_t(None) = {repr(ovx_string_t(None))}")
    assert ovx_string_t(None).ptr is not None
    assert ovx_string_t(None).length == 0
    assert str(ovx_string_t(None)) == ""

    print(f"ovx_string_t('') = {repr(ovx_string_t(''))}")
    assert ovx_string_t("").ptr is not None
    assert ovx_string_t("").length == 0
    assert str(ovx_string_t("")) == ""

    test_string = "Hello, ovx_string_t!"
    print(f"ovx_string_t('{test_string}') = {repr(ovx_string_t(test_string))}")
    s = ovx_string_t(test_string)
    assert s.ptr is not None
    assert s.length == len(test_string)
    assert str(s) == test_string

    assert str(ovx_string_t(0)) == "0" and ovx_string_t(0).length == 1
    assert str(ovx_string_t(False)) == "False" and ovx_string_t(False).length == 5
    assert str(ovx_string_t([])) == "[]" and ovx_string_t([]).length == 2


def test_dlpack_struct_layouts():
    """Verify DLPack ctypes struct layouts match expected C ABI.

    DLPack uses natural alignment (no #pragma pack). ctypes uses native
    alignment by default, which matches C compiler behavior on standard
    platforms. This test catches any platform-specific misalignment issues.
    """
    import ctypes

    from ovrtx._src.dlpack import (
        DLManagedTensor,
        DLManagedTensorVersioned,
        DLTensor,
    )

    # DLTensor (DLPack 0.8)
    assert ctypes.sizeof(DLTensor) == 48, "DLTensor size mismatch"
    assert DLTensor.data.offset == 0
    assert DLTensor.device.offset == 8
    assert DLTensor.ndim.offset == 16
    assert DLTensor.dtype.offset == 20
    assert DLTensor.shape.offset == 24
    assert DLTensor.strides.offset == 32
    assert DLTensor.byte_offset.offset == 40

    # DLManagedTensor (DLPack 0.8) - dl_tensor FIRST
    assert ctypes.sizeof(DLManagedTensor) == 64, "DLManagedTensor size mismatch"
    assert DLManagedTensor.dl_tensor.offset == 0
    assert DLManagedTensor.manager_ctx.offset == 48
    assert DLManagedTensor.deleter.offset == 56

    # DLManagedTensorVersioned (DLPack 1.0) - dl_tensor LAST
    assert ctypes.sizeof(DLManagedTensorVersioned) == 80, "DLManagedTensorVersioned size mismatch"
    assert DLManagedTensorVersioned.version.offset == 0
    assert DLManagedTensorVersioned.manager_ctx.offset == 8
    assert DLManagedTensorVersioned.deleter.offset == 16
    assert DLManagedTensorVersioned.flags.offset == 24
    assert DLManagedTensorVersioned.dl_tensor.offset == 32


@pytest.mark.usd_scene("cube.usda")
def test_reset(renderer, usd_scene):
    """Test reset() clears sensor simulation history."""
    usd_handle = renderer.add_usd(str(usd_scene))
    assert usd_handle is not None, "USD handle should be valid"

    renderer.step(render_products={"ovrtx_debug_dump_stage"}, delta_time=0.1)
    renderer.reset(time=0.0)
    renderer.step(render_products={"ovrtx_debug_dump_stage"}, delta_time=0.1)

    print("[PASS] reset test passed: sensor history reset without error")


@pytest.mark.usd_scene("cube.usda")
def test_reset_stage(renderer, usd_scene, test_output_dir):
    """Test reset_stage() clears USD content from runtime stage."""
    usd_handle = renderer.add_usd(str(usd_scene))
    assert usd_handle is not None, "USD handle should be valid"

    dump_before = get_stage_debug_dump(renderer)
    assert 'Mesh "Cube"' in dump_before, "Cube should exist before reset"
    (test_output_dir / "test_reset_stage_before.usda").write_text(dump_before, encoding="utf-8")

    renderer.reset_stage()

    dump_after = get_stage_debug_dump(renderer)
    (test_output_dir / "test_reset_stage_after.usda").write_text(dump_after, encoding="utf-8")
    assert 'Mesh "Cube"' not in dump_after, "Cube should not exist after reset_stage"
    assert "/World" not in dump_after, "/World should not exist after reset_stage"

    print("[PASS] reset_stage test passed: stage cleared")


@pytest.mark.usd_scene("cube.usda")
def test_add_usd_layer(renderer, usd_scene, test_output_dir):
    """Test add_usd_layer() - scene first (root), inline layer second (with path_prefix)."""
    # 1. Load scene FIRST as root layer
    usd_handle = renderer.add_usd(str(usd_scene))
    assert usd_handle is not None, "USD handle should be valid after loading"

    # 2. Add inline layer SECOND with path_prefix (reference composition)
    # Note: Fabric dump shows fabric attributes, not USD xformOps from referenced layers
    layer_handle = renderer.add_usd_layer(
        """#usda 1.0
(
    defaultPrim = "Layer"
)

def Xform "Layer" {
    matrix4d xformOp:transform = ( (1, 0, 0, 0), (0, 1, 0, 0), (0, 0, 1, 0), (999, 888, 777, 1) )
    uniform token[] xformOpOrder = ["xformOp:transform"]
}
""",
        path_prefix="/Layer",
    )
    assert layer_handle is not None, "Layer handle should be valid after loading inline content"

    dump = get_stage_debug_dump(renderer)
    (test_output_dir / "test_add_usd_layer.usda").write_text(dump, encoding="utf-8")

    # Verify /Layer prim was created with composed transform from inline layer
    assert 'Xform "Layer"' in dump, "/Layer Xform should exist"
    assert "999" in dump and "888" in dump and "777" in dump, "Layer transform (999,888,777) should be composed"
    # Verify scene cube exists (root layer)
    assert 'Mesh "Cube"' in dump, "Scene cube should exist"

    print("[PASS] add_usd_layer test passed: /Layer with composed transform verified")


@pytest.mark.usd_scene("cube.usda")
def test_remove_usd(renderer, usd_scene, test_output_dir):
    """Test remove_usd() removes specific USD content."""
    usd_handle = renderer.add_usd(str(usd_scene))
    assert usd_handle is not None, "USD handle should be valid"

    dump_before = get_stage_debug_dump(renderer)
    assert 'Mesh "Cube"' in dump_before, "Cube should exist before remove"
    (test_output_dir / "test_remove_usd_before.usda").write_text(dump_before, encoding="utf-8")

    renderer.remove_usd(usd_handle)

    dump_after = get_stage_debug_dump(renderer)
    (test_output_dir / "test_remove_usd_after.usda").write_text(dump_after, encoding="utf-8")
    assert 'Mesh "Cube"' not in dump_after, "Cube should not exist after remove_usd"

    print("[PASS] remove_usd test passed: content removed")


@pytest.mark.usd_scene("simple_scene_animated.usda")
def test_update_from_usd_time(renderer, usd_scene, test_output_dir):
    """Test update_from_usd_time() evaluates time-sampled attributes."""
    usd_handle = renderer.add_usd(str(usd_scene))
    assert usd_handle is not None, "USD handle should be valid"

    renderer.update_from_usd_time(0.0)
    dump_t0 = get_stage_debug_dump(renderer)
    (test_output_dir / "test_update_from_usd_time_t0.usda").write_text(dump_t0, encoding="utf-8")
    assert "100" in dump_t0, "Cube should be at initial position at t=0"

    renderer.update_from_usd_time(1.0)
    dump_t1 = get_stage_debug_dump(renderer)
    (test_output_dir / "test_update_from_usd_time_t1.usda").write_text(dump_t1, encoding="utf-8")

    assert dump_t0 != dump_t1, "Stage should differ between t=0 and t=1"
    print("[PASS] update_from_usd_time test passed: time-sampled attributes updated")


@pytest.mark.usd_scene("cube.usda")
def test_clone_usd(renderer, usd_scene, test_output_dir):
    """Test clone_usd() clones a prim to multiple targets."""
    usd_handle = renderer.add_usd(str(usd_scene))
    assert usd_handle is not None, "USD handle should be valid"

    renderer.clone_usd("/World/Cube", ["/World/Cube1", "/World/Cube2"])

    dump = get_stage_debug_dump(renderer)
    (test_output_dir / "test_clone_usd.usda").write_text(dump, encoding="utf-8")

    assert 'Mesh "Cube"' in dump, "Original Cube should exist"
    assert 'Mesh "Cube1"' in dump, "Clone Cube1 should exist"
    assert 'Mesh "Cube2"' in dump, "Clone Cube2 should exist"

    print("[PASS] clone_usd test passed: prim cloned to multiple targets")


@pytest.mark.usd_scene("cube.usda")
def test_write_attribute(renderer, usd_scene, test_output_dir):
    """Test write_attribute writes matrix data to a prim attribute."""
    from ovrtx.math import Matrix4d

    # Load cube.usda test scene
    usd_handle = renderer.add_usd(str(usd_scene))
    assert usd_handle is not None, "USD handle should be valid after loading"

    # Create Matrix4d (identity matrix with [0][0] = 99.0)
    matrix = Matrix4d()
    matrix.SetIdentity()
    matrix[0][0] = 99.0

    # Write attribute using write_attribute (sync by default)
    renderer.write_attribute(
        prim_paths=["/World/Cube"],
        attribute_name="omni:fabric:worldMatrix",
        tensor=matrix.to_dltensor(),
        semantic="transform_4x4",
    )

    dump = get_stage_debug_dump(renderer)
    (test_output_dir / "test_write_attribute.usda").write_text(dump, encoding="utf-8")

    # Verify the matrix was written correctly
    assert "custom matrix4d omni:fabric:worldMatrix" in dump, "Debug dump should contain matrix attribute"
    assert (
        "( (99, 0, 0, 0), (0, 1, 0, 0), (0, 0, 1, 0), (0, 0, 0, 1) )" in dump
    ), "Debug dump should contain expected matrix values"

    print("[PASS] write_attribute test passed: Matrix written and verified via debug dump")


@pytest.mark.usd_scene("cube.usda")
def test_attribute_binding_single_write(renderer, usd_scene, test_output_dir):
    """Test write_attribute using binding handle for a single write operation."""
    from ovrtx.math import Matrix4d

    # Load cube.usda test scene
    usd_handle = renderer.add_usd(str(usd_scene))
    assert usd_handle is not None, "USD handle should be valid after loading"

    # Create matrix and get DLTensor (extract dtype from tensor)
    matrix = Matrix4d()
    matrix.SetIdentity()
    matrix[0][0] = 99.0
    dl_tensor = matrix.to_dltensor()

    # Create attribute binding once
    attr_binding = renderer.bind_attribute(
        prim_paths=["/World/Cube"], attribute_name="omni:fabric:worldMatrix", semantic="transform_4x4"
    )

    # Write using binding (instead of prim_paths/attribute_name)
    attr_binding.write(dl_tensor)

    dump = get_stage_debug_dump(renderer)
    (test_output_dir / "test_single_write_attribute_with_binding_handle.usda").write_text(dump, encoding="utf-8")

    assert "custom matrix4d omni:fabric:worldMatrix" in dump
    assert "( (99, 0, 0, 0), (0, 1, 0, 0), (0, 0, 1, 0), (0, 0, 0, 1) )" in dump

    # Cleanup
    attr_binding.unbind()

    print("[PASS] write_attribute with binding (single write) test passed")


@pytest.mark.usd_scene("cube.usda")
def test_attribute_binding_multiple_writes(renderer, usd_scene, test_output_dir):
    """Test write_attribute with persistent binding handle for multiple writes."""
    from ovrtx.math import Matrix4d

    # Load cube.usda test scene
    usd_handle = renderer.add_usd(str(usd_scene))
    assert usd_handle is not None, "USD handle should be valid after loading"

    # Create matrix and get DLTensor (extract dtype from tensor)
    matrix = Matrix4d()
    matrix.SetIdentity()
    matrix[0][0] = 99.0
    dl_tensor = matrix.to_dltensor()

    # Create attribute binding once (dtype inferred from semantic)
    attr_binding = renderer.bind_attribute(
        prim_paths=["/World/Cube"], attribute_name="omni:fabric:worldMatrix", semantic="transform_4x4"
    )

    # Reuse binding for multiple writes
    for value in [99.0, 11.0]:
        matrix = Matrix4d()
        matrix.SetIdentity()
        matrix[0][0] = value
        dl_tensor = matrix.to_dltensor()

        attr_binding.write(dl_tensor)

    # Cleanup
    attr_binding.unbind()

    dump = get_stage_debug_dump(renderer)
    (test_output_dir / "test_write_attribute_multiple_with_binding_handle.usda").write_text(dump, encoding="utf-8")

    # Verify last write (11.0)
    assert "custom matrix4d omni:fabric:worldMatrix" in dump
    assert "( (11, 0, 0, 0), (0, 1, 0, 0), (0, 0, 1, 0), (0, 0, 0, 1) )" in dump

    print("[PASS] write_attribute with binding handle (multiple writes) test passed")


@pytest.mark.usd_scene("cube.usda")
def test_write_array_attribute(renderer, usd_scene, test_output_dir):
    """Test writing array attributes with variable lengths per prim."""
    import numpy as np
    from ovrtx._src.dlpack import DLTensor

    renderer.add_usd(str(usd_scene))

    # Write array attribute to single prim
    face_counts_1 = np.array([99, 98, 97, 96, 95, 94], dtype=np.int32)
    renderer.write_array_attribute(
        prim_paths=["/World/Cube"],
        attribute_name="faceVertexCounts",
        tensors=[DLTensor.from_dlpack(face_counts_1)],
    )

    dump = get_stage_debug_dump(renderer)
    assert "[99, 98, 97, 96, 95, 94]" in dump

    # Clone prim and write different array sizes to multiple prims
    renderer.clone_usd("/World/Cube", ["/World/Cube3"])
    face_counts_2 = np.array([16, 17, 18], dtype=np.int32)
    renderer.write_array_attribute(
        prim_paths=["/World/Cube", "/World/Cube3"],
        attribute_name="faceVertexCounts",
        tensors=[DLTensor.from_dlpack(face_counts_1), DLTensor.from_dlpack(face_counts_2)],
        prim_mode="create_new",
    )

    dump = get_stage_debug_dump(renderer)
    (test_output_dir / "test_write_array_attribute.usda").write_text(dump, encoding="utf-8")
    assert "[99, 98, 97, 96, 95, 94]" in dump
    assert "[16, 17, 18]" in dump
    print("[PASS] write_array_attribute with variable lengths")


@pytest.mark.usd_scene("cube.usda")
def test_bind_array_attribute(renderer, usd_scene, test_output_dir):
    """Test persistent binding for array attribute writes."""
    import numpy as np
    from ovrtx._src.dlpack import DLDataType, DLTensor

    renderer.add_usd(str(usd_scene))
    renderer.clone_usd("/World/Cube", ["/World/Cube3"])

    # Create persistent binding for array attribute (int[] requires dtype)
    binding = renderer.bind_array_attribute(
        prim_paths=["/World/Cube", "/World/Cube3"],
        attribute_name="faceVertexCounts",
        dtype=DLDataType.from_str("int32"),
        prim_mode="create_new",
    )

    # First write - different lengths per prim
    face_counts_1 = np.array([10, 20, 30], dtype=np.int32)
    face_counts_2 = np.array([40, 50], dtype=np.int32)
    binding.write([DLTensor.from_dlpack(face_counts_1), DLTensor.from_dlpack(face_counts_2)])

    dump = get_stage_debug_dump(renderer)
    assert "[10, 20, 30]" in dump
    assert "[40, 50]" in dump

    # Second write (reuse binding, change lengths)
    face_counts_1 = np.array([100, 200], dtype=np.int32)
    face_counts_2 = np.array([300, 400, 500, 600], dtype=np.int32)
    binding.write([DLTensor.from_dlpack(face_counts_1), DLTensor.from_dlpack(face_counts_2)])

    dump = get_stage_debug_dump(renderer)
    (test_output_dir / "test_bind_array_attribute.usda").write_text(dump, encoding="utf-8")
    assert "[100, 200]" in dump
    assert "[300, 400, 500, 600]" in dump

    # Cleanup
    binding.unbind()

    print("[PASS] bind_array_attribute + multiple writes with varying lengths")


@pytest.mark.usd_scene("cube.usda")
@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_map_attribute(device: str, renderer, usd_scene, test_output_dir):
    """Test map/unmap workflow on CPU and CUDA.

    Maps an attribute, writes data via Matrix4d.to_dltensor() + wp.copy(),
    unmaps to apply changes, and verifies via debug dump.
    """
    import numpy as np
    from ovrtx.math import Matrix4d

    # Load cube.usda test scene
    usd_handle = renderer.add_usd(str(usd_scene))
    assert usd_handle is not None, "USD handle should be valid after loading"

    # Create source matrix: identity with [0][0] = 99.0
    source = Matrix4d()
    source.SetIdentity()
    source[0][0] = 99.0
    source_np = np.array(source.v, dtype=np.float64).reshape(4, 4)

    # Map attribute - use context manager for auto-unmap
    with renderer.map_attribute(
        prim_paths=["/World/Cube"],
        attribute_name="omni:fabric:worldMatrix",
        semantic="transform_4x4",
        device=device,
        device_id=0,
    ) as mapping:
        # Verify mapping is valid
        assert mapping is not None, "Mapping should be valid"
        assert mapping.map_handle != 0, "Map handle should be non-zero"

        if device == "cpu":
            # CPU: Direct NumPy assignment (zero-copy write)
            np_array = np.from_dlpack(mapping.tensor)
            assert np_array.shape == (1, 4, 4), f"Expected shape (1, 4, 4), got {np_array.shape}"
            np_array[:] = source_np
        else:
            # CUDA: Use wp.copy for CPU->GPU transfer
            source_wp = wp.from_numpy(source_np, dtype=wp.mat44d)
            dest_wp = wp.from_dlpack(mapping.tensor)
            assert dest_wp.shape == (1, 4, 4), f"Expected shape (1, 4, 4), got {dest_wp.shape}"
            wp.copy(dest=dest_wp, src=source_wp)
            wp.synchronize()  # Ensure transfer completes before unmap
    # Auto-unmapped here

    dump = get_stage_debug_dump(renderer)
    (test_output_dir / f"test_map_attribute_{device}.usda").write_text(dump, encoding="utf-8")

    # Verify the matrix was written correctly
    assert "custom matrix4d omni:fabric:worldMatrix" in dump, "Debug dump should contain matrix attribute"
    assert (
        "( (99, 0, 0, 0), (0, 1, 0, 0), (0, 0, 1, 0), (0, 0, 0, 1) )" in dump
    ), "Debug dump should contain expected matrix values"

    print(f"[PASS] map_attribute test passed on {device.upper()}: Matrix mapped, written via wp.copy, verified")


@pytest.mark.usd_scene("cube.usda")
@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_map_attribute_existing_only(device: str, renderer, usd_scene, test_output_dir):
    """Test prim_mode="existing_only" ignores missing prims.

    Maps two prims where one doesn't exist. With prim_mode="existing_only",
    the operation succeeds and only the existing prim is written.
    """
    import numpy as np
    from ovrtx.math import Matrix4d

    renderer.add_usd(str(usd_scene))

    # Create two source matrices with different [0][0] values
    matrix0 = Matrix4d()
    matrix0.SetIdentity()
    matrix0[0][0] = 99.0
    matrix0_np = np.array(matrix0.v, dtype=np.float64).reshape(4, 4)

    matrix1 = Matrix4d()
    matrix1.SetIdentity()
    matrix1[0][0] = 11.0  # This won't be used (prim missing)
    matrix1_np = np.array(matrix1.v, dtype=np.float64).reshape(4, 4)

    # Stack into (2, 4, 4) array
    source_np = np.stack([matrix0_np, matrix1_np], axis=0)

    # Map with one existing, one missing prim
    with renderer.map_attribute(
        prim_paths=["/World/Cube", "/World/Cube1"],  # Cube1 doesn't exist
        attribute_name="omni:fabric:worldMatrix",
        semantic="transform_4x4",
        prim_mode="existing_only",  # Should succeed, ignoring missing
        device=device,
        device_id=0,
    ) as mapping:
        assert mapping is not None, "Mapping should be valid"
        # Shape: (2, 4, 4) - two slots allocated even though second prim missing

        if device == "cpu":
            np_array = np.from_dlpack(mapping.tensor)
            assert np_array.shape == (2, 4, 4), f"Expected shape (2, 4, 4), got {np_array.shape}"
            np_array[:] = source_np
        else:
            source_wp = wp.from_numpy(source_np, dtype=wp.mat44d)
            dest_wp = wp.from_dlpack(mapping.tensor)
            assert dest_wp.shape == (2, 4, 4), f"Expected shape (2, 4, 4), got {dest_wp.shape}"
            wp.copy(dest=dest_wp, src=source_wp)
            wp.synchronize()
    # Auto-unmapped here

    dump = get_stage_debug_dump(renderer)
    (test_output_dir / f"test_map_attribute_existing_only_{device}.usda").write_text(dump, encoding="utf-8")

    # Verify the first matrix (99.0) was written
    assert "( (99, 0, 0, 0), (0, 1, 0, 0), (0, 0, 1, 0), (0, 0, 0, 1) )" in dump

    print(f"[PASS] map_attribute_existing_only test passed on {device.upper()}")


@pytest.mark.usd_scene("cube.usda")
def test_map_attribute_must_exist_failure(renderer, usd_scene, test_output_dir):
    """Test prim_mode="must_exist" fails on missing prim.

    Maps a non-existing prim with prim_mode="must_exist".
    Map succeeds (deferred validation), but unmap should fail.
    """
    renderer.add_usd(str(usd_scene))

    # Map with must_exist on non-existing prim (don't use context manager - we test unmap failure)
    mapping = renderer.map_attribute(
        prim_paths=["/World/Cube1"],  # Doesn't exist
        attribute_name="omni:fabric:worldMatrix",
        semantic="transform_4x4",
        prim_mode="must_exist",  # Should fail on unmap
    )

    # Unmap should raise RuntimeError (works in both pytest and standalone execution)
    try:
        renderer.unmap_attribute(mapping)
        raise AssertionError("Expected RuntimeError but none was raised")
    except RuntimeError as e:
        # Verify error message indicates the prim was not found
        assert "Path/Attribute not found" in str(e), f"Unexpected error message: {e}"

    print("[PASS] map_attribute_must_exist_failure test passed")


@wp.kernel
def set_gpu_test_matrices(transforms: wp.array(dtype=wp.mat44d)):
    """Set two identity matrices with custom [0][0] and [3][3] values for GPU test."""
    _0 = wp.float64(0.0)
    _1 = wp.float64(1.0)
    # Matrix 0: [0][0]=777, [3][3]=888
    transforms[0] = wp.mat44d(
        wp.float64(777.0), _0, _0, _0, _0, _1, _0, _0, _0, _0, _1, _0, _0, _0, _0, wp.float64(888.0)
    )
    # Matrix 1: [0][0]=555, [3][3]=666
    transforms[1] = wp.mat44d(
        wp.float64(555.0), _0, _0, _0, _0, _1, _0, _0, _0, _0, _1, _0, _0, _0, _0, wp.float64(666.0)
    )


@pytest.mark.usd_scene("cube.usda")
def test_map_attribute_gpu(renderer, usd_scene, test_output_dir):
    """Test GPU map/unmap workflow with Warp kernel writes."""
    renderer.add_usd(str(usd_scene))

    # Create /World/Cube1 so we have two prims, map to GPU
    with renderer.map_attribute(
        prim_paths=["/World/Cube", "/World/Cube1"],
        attribute_name="omni:fabric:worldMatrix",
        semantic="transform_4x4",
        prim_mode="create_new",
        device="cuda",
        device_id=0,
    ) as mapping:
        # Shape: (2, 4, 4) due to semantic-aware reshaping
        assert mapping.tensor.shape == (2, 4, 4), f"Expected (2, 4, 4), got {mapping.tensor.shape}"

        # Write matrices directly on GPU via Warp kernel
        # Matrix 0: [0][0]=777, [3][3]=888
        # Matrix 1: [0][0]=555, [3][3]=666
        wp_transforms = wp.from_dlpack(mapping.tensor, dtype=wp.mat44d)
        wp.launch(kernel=set_gpu_test_matrices, dim=1, inputs=[wp_transforms], device="cuda")
        wp.synchronize()
    # Auto-unmapped here

    dump = get_stage_debug_dump(renderer)
    (test_output_dir / "test_map_attribute_gpu.usda").write_text(dump, encoding="utf-8")

    assert "( (777, 0, 0, 0), (0, 1, 0, 0), (0, 0, 1, 0), (0, 0, 0, 888) )" in dump
    assert "( (555, 0, 0, 0), (0, 1, 0, 0), (0, 0, 1, 0), (0, 0, 0, 666) )" in dump

    print("[PASS] map_attribute GPU test passed")


@pytest.mark.usd_scene("cube.usda")
@pytest.mark.parametrize("sync_mode", ["event", "stream"])
def test_unmap_attribute_cuda_sync(sync_mode: str, renderer, usd_scene, test_output_dir):
    """Test CUDA event/stream sync parameters for unmap_attribute.

    Tests that CUDA sync parameters (event/stream) are correctly passed through
    the Python API to the C library. The C library waits on the event/stream
    before consuming the mapped data, enabling asynchronous GPU workflows.

    Verifies that data written on a custom CUDA stream is correctly committed
    to Fabric when using event-based or stream-based synchronization.
    """
    renderer.add_usd(str(usd_scene))

    # Map to GPU with two prims (create_new mode creates /World/Cube1 if needed)
    # Same pattern as test_map_attribute_gpu which works
    with renderer.map_attribute(
        prim_paths=["/World/Cube", "/World/Cube1"],
        attribute_name="omni:fabric:worldMatrix",
        semantic="transform_4x4",
        device="cuda",
        prim_mode="create_new",
    ) as mapping:
        # Verify device property is set correctly
        assert mapping.device == "cuda", f"Expected device='cuda', got '{mapping.device}'"

        # Write via Warp kernel on custom stream
        # Create stream on the same device as the mapped tensor to avoid multi-GPU issues
        wp_transforms = wp.from_dlpack(mapping.tensor, dtype=wp.mat44d)
        stream = wp.Stream(device=wp_transforms.device)
        wp.launch(kernel=set_gpu_test_matrices, dim=1, inputs=[wp_transforms], stream=stream)

        # Unmap with CUDA sync - C library waits on event/stream before consuming data
        if sync_mode == "event":
            cuda_event = stream.record_event()
            mapping.unmap(event=cuda_event.cuda_event)
        else:
            mapping.unmap(stream=stream.cuda_stream)
    # __exit__ sees _unmapped=True -> no-op

    # Verify data was written correctly (proves sync worked)
    dump = get_stage_debug_dump(renderer)
    (test_output_dir / f"test_unmap_cuda_sync_{sync_mode}.usda").write_text(dump, encoding="utf-8")

    assert "( (777, 0, 0, 0), (0, 1, 0, 0), (0, 0, 1, 0), (0, 0, 0, 888) )" in dump
    assert "( (555, 0, 0, 0), (0, 1, 0, 0), (0, 0, 1, 0), (0, 0, 0, 666) )" in dump

    print(f"[PASS] unmap_attribute CUDA sync ({sync_mode}) test passed")


@pytest.mark.usd_scene("simple_scene.usda")
def test_renderer(renderer, usd_scene, test_output_dir):
    import numpy as np
    from PIL import Image

    # Note: Renderer creation/deletion is tested implicitly by session fixture lifecycle.
    # The fixture calls reset_stage() before each test, ensuring clean state.

    print(f"Successfully created renderer with config:\n{renderer.config}")

    # Load USD file
    usd_handle = renderer.add_usd(str(usd_scene))
    assert usd_handle is not None, "USD handle should be valid after loading"
    print(f"Successfully loaded USD file, handle: {usd_handle}")

    # Step renderer for 10 steps of 0.1 seconds each
    render_products = {
        "/Render/OmniverseKit/HydraTextures/ViewportTexture0",
        "/Render/OmniverseKit/HydraTextures/ViewportTexture1",
        "/Render/OmniverseKit/HydraTextures/ViewportTexture2",
    }

    for i in range(10):
        products = renderer.step(render_products=render_products, delta_time=0.1)
        assert products is not None, f"Step products should be valid at time {i * 0.1}s"

        print(f"{i * 0.1}s:")
        for product_name, product in products.items():
            print(f"  Product: {product_name}")

            for frame in product.frames:
                var_devices = {}  # Collect device info for each render var

                for var_name, render_var in frame.render_vars.items():
                    if var_name == "LdrColor":
                        filename_prefix = f"{product_name.split('/')[-1]}_{var_name}_{i:03d}"
                        with render_var.map(device="cpu") as mapping:
                            np_array = np.from_dlpack(mapping.tensor)
                            img = Image.fromarray(np_array)
                            filename = test_output_dir / f"{filename_prefix}.png"
                            img.save(filename)

                        for map_device in ["cpu", "cuda"]:
                            with render_var.map(device=map_device) as mapping:
                                wp_image = wp.from_dlpack(mapping.tensor, dtype=wp.vec4ub)
                                wp_output = wp.empty(wp_image.shape, dtype=wp.uint8, device=wp_image.device)
                                wp.launch(
                                    sobel_kernel,
                                    dim=wp_image.shape,
                                    inputs=[wp_image, wp_output],
                                    device=wp_image.device,
                                )
                                wp.synchronize_device(wp_image.device)
                                filename = test_output_dir / f"{filename_prefix}_sobel_{map_device}.png"
                                Image.fromarray(wp_output.numpy()).save(filename)

                                var_devices[f"{var_name}({map_device})"] = str(wp_image.device)

                # Print frame summary with device info
                devices_str = ", ".join(f"{k}={v}" for k, v in var_devices.items())
                print(f"    Frame: {frame.start_time:.2f}s - {frame.end_time:.2f}s, vars: [{devices_str}]")


@pytest.mark.usd_scene("simple_scene.usda")
@pytest.mark.parametrize("sync_mode", ["global", "event", "stream"])
def test_renderer_cuda(sync_mode: str, renderer, usd_scene, test_output_dir):
    """Test CUDA device mapping with different sync modes.

    - global: wp.synchronize_device() on default stream (baseline)
    - event: explicit CUDA event sync on unmap (custom stream)
    - stream: explicit CUDA stream sync on unmap (custom stream)

    Performs actual GPU work (copy) and verifies data integrity to prove
    synchronization worked correctly.
    """
    renderer.add_usd(str(usd_scene))

    render_products = {
        "/Render/OmniverseKit/HydraTextures/ViewportTexture0",
        "/Render/OmniverseKit/HydraTextures/ViewportTexture1",
        "/Render/OmniverseKit/HydraTextures/ViewportTexture2",
    }

    products = renderer.step(render_products=render_products, delta_time=0.1)

    device_ids = set()
    total_outputs = 0

    print(f"\nCUDA Mapping Test (sync_mode={sync_mode}):")
    for product_name, product in products.items():
        for frame in product.frames:
            for var_name, render_var in frame.render_vars.items():
                with render_var.map(device="cuda") as mapping:
                    wp_image = wp.from_dlpack(mapping.tensor)

                    # Validate CUDA device
                    assert "cuda" in str(wp_image.device), f"Expected CUDA device, got {wp_image.device}"
                    device_ids.add(str(wp_image.device))
                    total_outputs += 1

                    with wp.ScopedDevice(device=wp_image.device):
                        np_image = wp_image.numpy()  # obtain an early CPU copy of the source data before cloning on GPU

                        # Copy data over zeroed destination (using default or custom stream depending on sync mode)
                        stream = wp.get_stream() if sync_mode == "global" else wp.Stream()
                        with wp.ScopedStream(stream=stream, sync_exit=(sync_mode == "global")):
                            dest = wp.zeros_like(wp_image, device=wp_image.device)
                            wp.copy(dest, wp_image)
                            # ScopedScream will synchronize on device's default stream for sync_mode == "global"

                    # Exercise different explicit ovrtx sync modes
                    if sync_mode == "event":
                        cuda_event = stream.record_event()
                        mapping.unmap(event=cuda_event.cuda_event)
                    elif sync_mode == "stream":
                        mapping.unmap(stream=stream.cuda_stream)

                    # Verify exact copy (proves sync worked and full data transferred)
                    assert (dest.numpy() == np_image).all(), f"Copy should match source ({sync_mode})"

                print(f"  {var_name}: {wp_image.device} shape={dest.shape} sync={sync_mode}")

    print(f"  Total: {total_outputs} outputs on GPU(s): {sorted(device_ids)}")
    print(f"[PASS] CUDA mapping with {sync_mode} sync")


@pytest.mark.usd_scene("simple_scene.usda")
def test_renderer_warp(renderer, usd_scene, test_output_dir):
    """Test Warp interop with ovrtx rendering on both CPU and CUDA.

    Validates that the same render var can be mapped to different devices
    and that both devices see identical original data. Uses Warp kernels
    for computation (no NumPy dependency).
    """
    print("\nWarp Interop Test (CPU & CUDA):")

    # Define Warp kernel for computing sum over flat uint8 buffer
    @wp.kernel
    def compute_sum_kernel(data: wp.array(dtype=wp.uint8), result: wp.array(dtype=wp.float64)):
        i = wp.tid()
        pixel_value = wp.float64(data[i])  # Explicit float64 conversion
        wp.atomic_add(result, 0, pixel_value)

    def compute_sum_on_device(render_var, device: str, var_name: str) -> float:
        """Helper to map render var to device, launch Warp kernel, and compute sum."""
        with render_var.map(device=device) as mapping:
            wp_array = wp.from_dlpack(mapping.tensor)

            # Assert buffer is contiguous (no strides, direct memory access)
            assert wp_array.is_contiguous, "Buffer must be contiguous for efficient access"

            # Flatten creates a view (zero-copy) since buffer is contiguous
            wp_flat = wp_array.flatten()

            # Small result buffer for atomic accumulation (unavoidable - single output value)
            result = wp.zeros(1, dtype=wp.float64, device=wp_array.device)

            # Launch kernel - operates directly on mapped buffer
            wp.launch(compute_sum_kernel, dim=wp_flat.shape[0], inputs=[wp_flat, result], device=wp_array.device)

            # Synchronize to ensure kernel completes before reading result
            wp.synchronize_device(wp_array.device)

            computed_sum = float(result.numpy()[0])

            print(
                f"  {device.upper():4s}: {var_name} shape={wp_array.shape} dtype={wp_array.dtype} "
                f"device={wp_array.device} contiguous={wp_array.is_contiguous} sum={computed_sum:.1f}"
            )

            return computed_sum

    renderer.add_usd(str(usd_scene))

    # Step once
    render_products = {
        "/Render/OmniverseKit/HydraTextures/ViewportTexture0",
    }

    products = renderer.step(render_products=render_products, delta_time=0.1)

    for product_name, product in products.items():
        for frame in product.frames:
            for var_name, render_var in frame.render_vars.items():
                # Only test LdrColor to keep it lean
                if var_name != "LdrColor":
                    continue

                # Compute sum on both devices
                cuda_sum = compute_sum_on_device(render_var, "cuda", var_name)
                cpu_sum = compute_sum_on_device(render_var, "cpu", var_name)

                # Validate consistency with float tolerance
                sum_diff = abs(cuda_sum - cpu_sum)
                print(f"  Consistency: CUDA vs CPU sum diff = {sum_diff:.1f}")
                assert sum_diff < 1e-3, f"CUDA and CPU saw different data! Diff:{sum_diff:.1f}"


@pytest.mark.usd_scene("simple_scene.usda")
def test_step_async_polling(renderer, usd_scene):
    """Test async step with polling pattern - timeout then completion.

    Verifies that:
    - Immediate poll (timeout_ns=0) returns None without blocking
    - step_complete is False after timeout (operation still pending)
    - Subsequent poll can succeed after operation completes
    - step_complete is True after successful completion
    - Result is cached after completion (idempotent)
    """
    renderer.add_usd(str(usd_scene))

    render_products = {"/Render/OmniverseKit/HydraTextures/ViewportTexture0"}
    result = renderer.step_async(render_products=render_products, delta_time=1 / 60)

    # Phase 1: Immediate poll - should timeout (non-blocking)
    metadata = result.wait(step_timeout_ns=0)
    assert metadata is None, "Expected timeout on immediate poll"
    assert not result.step_complete, "step_complete must be False after timeout"

    # Phase 2: Poll again with sufficient timeout - should succeed
    metadata = result.wait(step_timeout_ns=10_000_000_000)  # 10 sec
    assert metadata is not None, "Polling must work after previous timeout"
    assert result.step_complete, "step_complete must be True after completion"

    # Phase 3: Idempotent - subsequent waits return cached result
    metadata2 = result.wait()
    assert metadata2 is metadata, "Should return cached result after completion"

    print("[PASS] test_step_async_polling")


def test_log_configuration(shared_renderer, test_output_dir):
    """Test that log file path and log level configuration are applied correctly.

    Uses shared_renderer fixture to ensure renderer is initialized before checking log files.
    """
    log_file = test_output_dir / "test_ovrtx.log"
    assert log_file.exists(), f"Log file should exist at {log_file}"
    assert log_file.stat().st_size > 0, "Log file should not be empty"

    print("[PASS] test_log_configuration")


def test_raw_library():
    """Test raw library bindings directly (no fixtures, does its own init/shutdown).

    This test must run last as it manipulates library ref-counting directly.
    """
    import ctypes

    from ovrtx._src.bindings import OVRTX_API_SUCCESS, _ovrtx_loader

    ovrtx_config = Renderer._to_c_config(OVRTX_TEST_CONFIG)
    bindings = _ovrtx_loader.create_bindings(ovrtx_config)
    assert bindings is not None, "Bindings are not valid after attempting to load ovrtx library."

    # Test ref-counting: ovrtx_initialize() can be called multiple times (increases ref-count)
    res = bindings._lib.ovrtx_initialize(ctypes.byref(ovrtx_config))
    assert (
        res.status == OVRTX_API_SUCCESS
    ), f"Re-initializing library should succeed (ref-counting): {bindings.get_last_error()}"

    # Balance the ref-count by calling shutdown
    res = bindings._lib.ovrtx_shutdown()
    assert (
        res.status == OVRTX_API_SUCCESS
    ), f"Shutdown after ref-count increase should succeed: {bindings.get_last_error()}"


if __name__ == "__main__":
    # Direct execution for quick testing during development
    # For full test runs, use: pytest

    import os

    TEST_OUTPUT_DIR.mkdir(exist_ok=True)

    # Helper to resolve scene paths for direct execution (bypasses pytest fixtures)
    def get_scene(name: str) -> Path:
        env_path = os.environ.get("TEST_DATA_ROOT")
        assert env_path, "TEST_DATA_ROOT environment variable must be set"
        scene_path = (Path(env_path) / name).resolve()
        assert scene_path.exists(), f"USD scene file not found: {scene_path}"
        return scene_path

    # Create shared renderer (mirrors session-scoped fixture)
    renderer = Renderer(OVRTX_TEST_CONFIG)
    assert renderer, "Renderer should be valid after creation"

    def run_test(test_func, *args):
        """Run a test with the shared renderer, resetting stage first."""
        renderer.reset_stage()
        renderer.reset(time=0.0)
        test_func(*args)

    # Tests without renderer
    test_ovx_string_t()
    test_dlpack_struct_layouts()

    # Tests with renderer (reset_stage and reset called before each)
    run_test(test_reset, renderer, get_scene("cube.usda"))
    run_test(test_reset_stage, renderer, get_scene("cube.usda"), TEST_OUTPUT_DIR)
    run_test(test_add_usd_layer, renderer, get_scene("cube.usda"), TEST_OUTPUT_DIR)
    run_test(test_remove_usd, renderer, get_scene("cube.usda"), TEST_OUTPUT_DIR)
    run_test(test_update_from_usd_time, renderer, get_scene("simple_scene_animated.usda"), TEST_OUTPUT_DIR)
    run_test(test_clone_usd, renderer, get_scene("cube.usda"), TEST_OUTPUT_DIR)
    run_test(test_write_attribute, renderer, get_scene("cube.usda"), TEST_OUTPUT_DIR)
    run_test(test_attribute_binding_single_write, renderer, get_scene("cube.usda"), TEST_OUTPUT_DIR)
    run_test(test_attribute_binding_multiple_writes, renderer, get_scene("cube.usda"), TEST_OUTPUT_DIR)
    run_test(test_write_array_attribute, renderer, get_scene("cube.usda"), TEST_OUTPUT_DIR)
    run_test(test_bind_array_attribute, renderer, get_scene("cube.usda"), TEST_OUTPUT_DIR)
    run_test(test_map_attribute, "cpu", renderer, get_scene("cube.usda"), TEST_OUTPUT_DIR)
    run_test(test_map_attribute, "cuda", renderer, get_scene("cube.usda"), TEST_OUTPUT_DIR)
    run_test(test_map_attribute_existing_only, "cpu", renderer, get_scene("cube.usda"), TEST_OUTPUT_DIR)
    run_test(test_map_attribute_existing_only, "cuda", renderer, get_scene("cube.usda"), TEST_OUTPUT_DIR)
    run_test(test_map_attribute_must_exist_failure, renderer, get_scene("cube.usda"), TEST_OUTPUT_DIR)
    run_test(test_map_attribute_gpu, renderer, get_scene("cube.usda"), TEST_OUTPUT_DIR)
    for sync_mode in ["event", "stream"]:
        run_test(test_unmap_attribute_cuda_sync, sync_mode, renderer, get_scene("cube.usda"), TEST_OUTPUT_DIR)
    run_test(test_renderer, renderer, get_scene("simple_scene.usda"), TEST_OUTPUT_DIR)
    for sync_mode in ["global", "event", "stream"]:
        run_test(test_renderer_cuda, sync_mode, renderer, get_scene("simple_scene.usda"), TEST_OUTPUT_DIR)
    run_test(test_renderer_warp, renderer, get_scene("simple_scene.usda"), TEST_OUTPUT_DIR)
    run_test(test_step_async_polling, renderer, get_scene("simple_scene.usda"))
    run_test(test_log_configuration, renderer, TEST_OUTPUT_DIR)

    # Cleanup renderer BEFORE test_raw_library (ensures proper Carbonite shutdown)
    del renderer

    # test_raw_library runs last - does its own init/shutdown
    test_raw_library()

    print("\n[PASS] All tests passed")

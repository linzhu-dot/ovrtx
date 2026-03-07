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

import numpy as np
import pytest
import warp as wp

from ovrtx import DataAccess, Device, PrimMode, Renderer, RendererConfig, Semantic

wp.init()


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
    with frame.render_vars["debug"].map(device=Device.CPU) as mapping:
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

    print("[PASS] ovx_string_t test passed")


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

    print("[PASS] DLPack struct layouts test passed")


@pytest.mark.parametrize("source", ["NumPy", "Warp"])
def test_from_dlpack_no_leak(source):
    """DLTensor.from_dlpack must not accumulate references to the source object."""
    import gc
    import sys

    from ovrtx._src.dlpack import DLTensor

    arr = np.ones((3, 4), dtype=np.float32) if source.lower() == "numpy" else wp.ones((3, 4), dtype=wp.float32)
    baseline = sys.getrefcount(arr)
    tensors = [DLTensor.from_dlpack(arr) for _ in range(1000)]
    held = sys.getrefcount(arr)
    assert (
        held > baseline
    ), f"from_dlpack should hold a ref to {source}: refcount {held} == baseline {baseline}"

    del tensors
    gc.collect()
    final = sys.getrefcount(arr)
    assert (
        final == baseline
    ), f"from_dlpack leaked {final - baseline} ref(s) to {source}: expected {baseline}, got {final}"


def test_version(renderer):
    """Test that version property returns a 3-tuple matching __version__."""

    from ovrtx import __version__ as ovrtx_version

    version = renderer.version
    assert isinstance(version, tuple) and len(version) == 3, f"Expected (major, minor, patch), got {version}"
    major, minor, patch = version
    assert all(isinstance(v, int) for v in version), f"Version components must be ints, got {version}"
    assert ovrtx_version.startswith(
        f"{major}.{minor}."
    ), f"Library version {major}.{minor}.{patch} does not match Python bindings {ovrtx_version}"

    version_md = Path(__file__).parent.parent / "VERSION.md"
    if version_md.exists():
        file_version = version_md.read_text().strip()
        c_version = f"{major}.{minor}.{patch}"
        assert file_version == c_version, f"VERSION.md says {file_version!r} but C API returned {c_version!r}"
    print(
        f"[PASS] version test passed: {major}.{minor}.{patch}"
        + (" (matches VERSION.md)" if version_md.exists() else "")
    )


@pytest.mark.usd_scene("cube.usda")
def test_reset(renderer, usd_scene):
    """Test reset() clears sensor simulation history."""

    usd_handle = renderer.add_usd(str(usd_scene))
    assert usd_handle is not None, f"USD handle should be valid after loading {usd_scene}"

    renderer.step(render_products={"ovrtx_debug_dump_stage"}, delta_time=0.1)
    renderer.reset(time=0.0)
    renderer.step(render_products={"ovrtx_debug_dump_stage"}, delta_time=0.1)

    print("[PASS] reset test passed: sensor history reset without error")


@pytest.mark.usd_scene("cube.usda")
def test_reset_stage(renderer, usd_scene, test_output_dir):
    """Test reset_stage() clears USD content from runtime stage."""

    usd_handle = renderer.add_usd(str(usd_scene))
    assert usd_handle is not None, f"USD handle should be valid after loading {usd_scene}"

    dump_before = get_stage_debug_dump(renderer)
    assert 'Mesh "Cube"' in dump_before, "Cube should exist before reset"
    (test_output_dir / "test_reset_stage_before.usda").write_text(dump_before, encoding="utf-8")

    # [snippet:reset-stage]
    renderer.reset_stage()
    # [/snippet:reset-stage]

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
    assert usd_handle is not None, f"USD handle should be valid after loading {usd_scene}"

    # 2. Add inline layer SECOND with path_prefix (reference composition)
    # Note: Fabric dump shows fabric attributes, not USD xformOps from referenced layers
    # [snippet:add-usd-layer]
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
    # [/snippet:add-usd-layer]
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
    assert usd_handle is not None, f"USD handle should be valid after loading {usd_scene}"

    dump_before = get_stage_debug_dump(renderer)
    assert 'Mesh "Cube"' in dump_before, "Cube should exist before remove"
    (test_output_dir / "test_remove_usd_before.usda").write_text(dump_before, encoding="utf-8")

    # [snippet:remove-usd]
    renderer.remove_usd(usd_handle)
    # [/snippet:remove-usd]

    dump_after = get_stage_debug_dump(renderer)
    (test_output_dir / "test_remove_usd_after.usda").write_text(dump_after, encoding="utf-8")
    assert 'Mesh "Cube"' not in dump_after, "Cube should not exist after remove_usd"

    print("[PASS] remove_usd test passed: content removed")


@pytest.mark.usd_scene("simple_scene_animated.usda")
def test_update_from_usd_time(renderer, usd_scene, test_output_dir):
    """Test update_from_usd_time() evaluates time-sampled attributes."""

    usd_handle = renderer.add_usd(str(usd_scene))
    assert usd_handle is not None, f"USD handle should be valid after loading {usd_scene}"

    # [snippet:update-from-usd-time]
    renderer.update_from_usd_time(0.0)
    # [/snippet:update-from-usd-time]
    dump_t0 = get_stage_debug_dump(renderer)
    (test_output_dir / "test_update_from_usd_time_t0.usda").write_text(dump_t0, encoding="utf-8")
    assert "100" in dump_t0, "Cube should be at initial position at t=0"

    # [snippet:update-from-usd-time]
    renderer.update_from_usd_time(1.0)
    # [/snippet:update-from-usd-time]
    dump_t1 = get_stage_debug_dump(renderer)
    (test_output_dir / "test_update_from_usd_time_t1.usda").write_text(dump_t1, encoding="utf-8")

    assert dump_t0 != dump_t1, "Stage should differ between t=0 and t=1"
    print("[PASS] update_from_usd_time test passed: time-sampled attributes updated")


@pytest.mark.usd_scene("cube.usda")
def test_clone_usd(renderer, usd_scene, test_output_dir):
    """Test clone_usd() clones a prim to multiple targets."""

    usd_handle = renderer.add_usd(str(usd_scene))
    assert usd_handle is not None, f"USD handle should be valid after loading {usd_scene}"

    # [snippet:clone-usd]
    renderer.clone_usd("/World/Cube", ["/World/Cube1", "/World/Cube2"])
    # [/snippet:clone-usd]

    dump = get_stage_debug_dump(renderer)
    (test_output_dir / "test_clone_usd.usda").write_text(dump, encoding="utf-8")

    assert 'Mesh "Cube"' in dump, "Original Cube should exist"
    assert 'Mesh "Cube1"' in dump, "Clone Cube1 should exist"
    assert 'Mesh "Cube2"' in dump, "Clone Cube2 should exist"

    print("[PASS] clone_usd test passed: prim cloned to multiple targets")


@pytest.mark.usd_scene("cube.usda")
@pytest.mark.parametrize(
    "device, sync_mode, data_access, use_async_api",
    [
        pytest.param("cpu", None, None, False, id="cpu-default"),
        pytest.param("cpu", None, DataAccess.SYNC, True, id="cpu-sync_data_access-async_api"),
        pytest.param("cpu", None, DataAccess.ASYNC, True, id="cpu-async_data_access-async_api"),
        pytest.param("cpu", None, DataAccess.ASYNC, False, id="cpu-async_data_access-sync_api"),
        pytest.param("cuda", "event", DataAccess.ASYNC, True, id="cuda_event-async_data_access-async_api"),
        pytest.param("cuda", "event", DataAccess.ASYNC, False, id="cuda_event-async_data_access-sync_api"),
        pytest.param("cuda", "stream", DataAccess.ASYNC, True, id="cuda_stream-async_data_access-async_api"),
        pytest.param("cuda", "stream", DataAccess.ASYNC, False, id="cuda_stream-async_data_access-sync_api"),
    ],
)
def test_write_attribute(device, sync_mode, data_access, use_async_api, renderer, usd_scene, test_output_dir):
    """Test write_attribute through the direct (non-binding) path on CPU and GPU.

    CPU: NumPy tensor, no CUDA sync.
    GPU: Warp tensor via wp.from_numpy on custom stream, CUDA stream/event sync.
    Covers default, explicit sync, and async data_access modes via both
    the sync and async Python API entry points.

    Note: GPU variants only test async data_access because the C API rejects
    OVRTX_DATA_ACCESS_SYNC for GPU buffers.
    """

    usd_handle = renderer.add_usd(str(usd_scene))
    assert usd_handle is not None, f"USD handle should be valid after loading {usd_scene}"

    # Build source matrix: identity with [0][0] = 99.0
    # [snippet:write-attribute-cpu]
    source_np = np.eye(4, dtype=np.float64)
    source_np[0][0] = 99.0
    source_np = source_np.reshape(1, 4, 4)  # (N, 4, 4) tensor format for single prim

    if device == "cpu":
        tensor = source_np
        cuda_kwargs = {}
    else:
        tensor = wp.from_numpy(source_np, dtype=wp.mat44d, device="cuda:0")
        wp_stream = tensor.device.stream
        if sync_mode == "event":
            cuda_event = wp_stream.record_event()
            cuda_kwargs = {"cuda_event": cuda_event.cuda_event}
        else:
            cuda_kwargs = {"cuda_stream": wp_stream.cuda_stream}

    write_kwargs = {
        "prim_paths": ["/World/Cube"],
        "attribute_name": "omni:xform",
        "tensor": tensor,
        **cuda_kwargs,
    }
    if data_access is not None:
        write_kwargs["data_access"] = data_access

    if use_async_api:
        op = renderer.write_attribute_async(**write_kwargs)
        if data_access == DataAccess.ASYNC:
            assert hasattr(op, "_storage_refs"), "Operation should have _storage_refs for async access"
            assert len(op._storage_refs) == 2, "Should have input_storage + binding_storage"
        op.wait()
    else:
        renderer.write_attribute(**write_kwargs)
    # [/snippet:write-attribute-cpu]

    dump = get_stage_debug_dump(renderer)
    da_tag = f"{data_access.name.lower()}_data_access" if data_access is not None else "default"
    api_tag = "async_api" if use_async_api else "sync_api"
    tag = f"{device}_{sync_mode or 'nosync'}-{da_tag}-{api_tag}"
    (test_output_dir / f"test_write_attribute_{tag}.usda").write_text(dump, encoding="utf-8")

    # Verify the matrix was written correctly
    assert "custom matrix4d omni:fabric:localMatrix" in dump, "Debug dump should contain matrix attribute"
    assert (
        "( (99, 0, 0, 0), (0, 1, 0, 0), (0, 0, 1, 0), (0, 0, 0, 1) )" in dump
    ), "Debug dump should contain expected matrix values"

    print(f"[PASS] write_attribute ({tag}) test passed")


@pytest.mark.usd_scene("cube.usda")
@pytest.mark.parametrize(
    "device, sync_mode, data_access, use_async_api",
    [
        pytest.param("cpu", None, None, False, id="cpu-default"),
        pytest.param("cpu", None, DataAccess.SYNC, True, id="cpu-sync_data_access-async_api"),
        pytest.param("cpu", None, DataAccess.ASYNC, True, id="cpu-async_data_access-async_api"),
        pytest.param("cpu", None, DataAccess.ASYNC, False, id="cpu-async_data_access-sync_api"),
        pytest.param("cuda", "event", DataAccess.ASYNC, True, id="cuda_event-async_data_access-async_api"),
        pytest.param("cuda", "event", DataAccess.ASYNC, False, id="cuda_event-async_data_access-sync_api"),
        pytest.param("cuda", "stream", DataAccess.ASYNC, True, id="cuda_stream-async_data_access-async_api"),
        pytest.param("cuda", "stream", DataAccess.ASYNC, False, id="cuda_stream-async_data_access-sync_api"),
    ],
)
def test_write_attribute_binding(device, sync_mode, data_access, use_async_api, renderer, usd_scene, test_output_dir):
    """Binding path write on CPU and GPU.

    CPU: NumPy tensor, no CUDA sync.
    GPU: Warp tensor via wp.from_numpy on custom stream, CUDA stream/event sync.
    Covers default, explicit sync, and async data_access modes via both
    the sync and async Python API entry points.

    Note: Only valid combinations are tested.
    """

    # Load cube.usda test scene
    usd_handle = renderer.add_usd(str(usd_scene))
    assert usd_handle is not None, f"USD handle should be valid after loading {usd_scene}"

    # Build 4x4 source matrices (shared between CPU and GPU paths)
    m0 = np.eye(4, dtype=np.float64)
    m0[0][0], m0[3][3] = 777.0, 888.0
    m1 = np.eye(4, dtype=np.float64)
    m1[0][0], m1[3][3] = 555.0, 666.0
    source_np = np.stack([m0, m1])

    # [snippet:bind-attribute-write]
    attr_binding = renderer.bind_attribute(
        prim_paths=["/World/Cube", "/World/Cube1"],
        attribute_name="omni:xform",
        semantic=Semantic.XFORM_MAT4x4,
        prim_mode=PrimMode.CREATE_NEW,
    )

    if device == "cpu":
        tensor = source_np
        cuda_kwargs = {}
    else:
        tensor = wp.from_numpy(source_np, dtype=wp.mat44d, device="cuda:0")
        wp_stream = tensor.device.stream
        if sync_mode == "event":
            cuda_event = wp_stream.record_event()
            cuda_kwargs = {"cuda_event": cuda_event.cuda_event}
        else:
            cuda_kwargs = {"cuda_stream": wp_stream.cuda_stream}

    write_kwargs = {**cuda_kwargs}
    if data_access is not None:
        write_kwargs["data_access"] = data_access

    if use_async_api:
        op = attr_binding.write_async(tensor, **write_kwargs)
        if data_access == DataAccess.ASYNC:
            assert hasattr(op, "_storage_refs"), "Operation should have _storage_refs for async access"
            assert len(op._storage_refs) == 1, "Binding path should only stash input_storage"
        op.wait()
    else:
        attr_binding.write(tensor, **write_kwargs)
    # [/snippet:bind-attribute-write]

    attr_binding.unbind()

    dump = get_stage_debug_dump(renderer)
    da_tag = f"{data_access.name.lower()}_data_access" if data_access is not None else "default"
    api_tag = "async_api" if use_async_api else "sync_api"
    tag = f"{device}_{sync_mode or 'nosync'}-{da_tag}-{api_tag}"
    (test_output_dir / f"test_write_attribute_binding_{tag}.usda").write_text(dump, encoding="utf-8")

    assert (
        "( (777, 0, 0, 0), (0, 1, 0, 0), (0, 0, 1, 0), (0, 0, 0, 888) )" in dump
    ), "Debug dump should contain expected matrix values for Cube"
    assert (
        "( (555, 0, 0, 0), (0, 1, 0, 0), (0, 0, 1, 0), (0, 0, 0, 666) )" in dump
    ), "Debug dump should contain expected matrix values for Cube1"

    print(f"[PASS] binding.write ({tag}) test passed")


@pytest.mark.usd_scene("cube.usda")
def test_attribute_binding_multiple_writes(renderer, usd_scene, test_output_dir):
    """Test write_attribute with persistent binding handle for multiple writes."""

    # Load cube.usda test scene
    usd_handle = renderer.add_usd(str(usd_scene))
    assert usd_handle is not None, f"USD handle should be valid after loading {usd_scene}"

    # [snippet:bind-attribute-multiple-writes]
    attr_binding = renderer.bind_attribute(
        prim_paths=["/World/Cube"], attribute_name="omni:xform", dtype=np.float64, shape=(4, 4)
    )

    # Reuse binding for multiple writes - pass NumPy arrays directly
    for value in [99.0, 11.0]:
        matrix = np.eye(4, dtype=np.float64)
        matrix[0][0] = value
        attr_binding.write(matrix.reshape(1, 4, 4))

    # Cleanup
    attr_binding.unbind()
    # [/snippet:bind-attribute-multiple-writes]

    dump = get_stage_debug_dump(renderer)
    (test_output_dir / "test_write_attribute_multiple_with_binding_handle.usda").write_text(dump, encoding="utf-8")

    # Verify last write (11.0)
    assert "custom matrix4d omni:fabric:localMatrix" in dump
    assert "( (11, 0, 0, 0), (0, 1, 0, 0), (0, 0, 1, 0), (0, 0, 0, 1) )" in dump

    # [snippet:map-binding-cpu]
    # Bindings also support zero-copy mapping (more efficient for repeated map/unmap cycles)
    map_binding = renderer.bind_attribute(
        prim_paths=["/World/Cube"], attribute_name="omni:xform", dtype=np.float64, shape=(4, 4)
    )

    with map_binding.map(device=Device.CPU) as mapping:
        array = np.from_dlpack(mapping.tensor)
        array[:] = np.eye(4, dtype=np.float64)

    map_binding.unbind()
    # [/snippet:map-binding-cpu]

    dump = get_stage_debug_dump(renderer)
    assert "custom matrix4d omni:fabric:localMatrix" in dump

    print("[PASS] write_attribute with binding handle (multiple writes) test passed")


@pytest.mark.usd_scene("cube.usda")
@pytest.mark.parametrize(
    "data_access, use_async_api",
    [
        pytest.param(None, False, id="cpu-default"),
        pytest.param(DataAccess.SYNC, True, id="cpu-sync_data_access-async_api"),
        pytest.param(DataAccess.ASYNC, True, id="cpu-async_data_access-async_api"),
        pytest.param(DataAccess.ASYNC, False, id="cpu-async_data_access-sync_api"),
    ],
)
def test_write_array_attribute(data_access, use_async_api, renderer, usd_scene, test_output_dir):
    """Test write_array_attribute through the direct (non-binding) path.

    Writes variable-length int32 arrays (faceVertexCounts) to two prims.
    Covers default, explicit sync, and async data_access modes via both
    the sync and async Python API entry points.

    Note: GPU variants are excluded because the C library rejects GPU memory for array attribute writes.
    """

    usd_handle = renderer.add_usd(str(usd_scene))
    assert usd_handle is not None, f"USD handle should be valid after loading {usd_scene}"
    renderer.clone_usd("/World/Cube", ["/World/Cube3"])

    # Build source arrays: different lengths per prim
    # [snippet:write-array-attribute]
    counts_1_np = np.array([99, 98, 97, 96, 95, 94], dtype=np.int32)
    counts_2_np = np.array([16, 17, 18], dtype=np.int32)

    write_kwargs = {
        "prim_paths": ["/World/Cube", "/World/Cube3"],
        "attribute_name": "faceVertexCounts",
        "tensors": [counts_1_np, counts_2_np],
        "prim_mode": PrimMode.CREATE_NEW,
    }
    if data_access is not None:
        write_kwargs["data_access"] = data_access

    if use_async_api:
        op = renderer.write_array_attribute_async(**write_kwargs)
        if data_access == DataAccess.ASYNC:
            assert hasattr(op, "_storage_refs"), "Operation should have _storage_refs for async access"
            assert len(op._storage_refs) == 2, "Should have input_storage + binding_storage"
        op.wait()
    else:
        renderer.write_array_attribute(**write_kwargs)
    # [/snippet:write-array-attribute]

    dump = get_stage_debug_dump(renderer)
    da_tag = f"{data_access.name.lower()}_data_access" if data_access is not None else "default"
    api_tag = "async_api" if use_async_api else "sync_api"
    tag = f"cpu-{da_tag}-{api_tag}"
    (test_output_dir / f"test_write_array_attribute_{tag}.usda").write_text(dump, encoding="utf-8")

    assert "[99, 98, 97, 96, 95, 94]" in dump, "Debug dump should contain first array values"
    assert "[16, 17, 18]" in dump, "Debug dump should contain second array values"

    print(f"[PASS] write_array_attribute ({tag}) test passed")


@pytest.mark.usd_scene("cube.usda")
@pytest.mark.parametrize(
    "data_access, use_async_api",
    [
        pytest.param(None, False, id="cpu-default"),
        pytest.param(DataAccess.SYNC, True, id="cpu-sync_data_access-async_api"),
        pytest.param(DataAccess.ASYNC, True, id="cpu-async_data_access-async_api"),
        pytest.param(DataAccess.ASYNC, False, id="cpu-async_data_access-sync_api"),
    ],
)
def test_bind_array_attribute(data_access, use_async_api, renderer, usd_scene, test_output_dir):
    """Test persistent binding for array attribute writes.

    Writes variable-length int32 arrays (faceVertexCounts) to two prims, then
    rewrites with different lengths to verify binding reuse across array sizes.
    Covers default, explicit sync, and async data_access modes via both
    the sync and async Python API entry points.

    Note: GPU variants are excluded because the C library rejects GPU memory for array attribute writes.
    """

    usd_handle = renderer.add_usd(str(usd_scene))
    assert usd_handle is not None, f"USD handle should be valid after loading {usd_scene}"
    renderer.clone_usd("/World/Cube", ["/World/Cube3"])

    # [snippet:bind-array-attribute]
    binding = renderer.bind_array_attribute(
        prim_paths=["/World/Cube", "/World/Cube3"],
        attribute_name="faceVertexCounts",
        dtype=np.int32,
        prim_mode=PrimMode.CREATE_NEW,
    )

    # --- First write ---
    counts_1_np = np.array([10, 20, 30], dtype=np.int32)
    counts_2_np = np.array([40, 50], dtype=np.int32)

    write_kwargs = {}
    if data_access is not None:
        write_kwargs["data_access"] = data_access

    if use_async_api:
        op = binding.write_async([counts_1_np, counts_2_np], **write_kwargs)
        if data_access == DataAccess.ASYNC:
            assert hasattr(op, "_storage_refs"), "Operation should have _storage_refs for async access"
            assert len(op._storage_refs) == 1, "Should have input_storage only (binding is persistent)"
        op.wait()
    else:
        binding.write([counts_1_np, counts_2_np], **write_kwargs)
    # [/snippet:bind-array-attribute]

    dump = get_stage_debug_dump(renderer)
    assert "[10, 20, 30]" in dump, "Debug dump should contain first write values (prim 1)"
    assert "[40, 50]" in dump, "Debug dump should contain first write values (prim 2)"

    # --- Second write (reuse binding, change lengths) ---
    counts_1_np = np.array([100, 200], dtype=np.int32)
    counts_2_np = np.array([300, 400, 500, 600], dtype=np.int32)

    if use_async_api:
        op = binding.write_async([counts_1_np, counts_2_np], **write_kwargs)
        op.wait()
    else:
        binding.write([counts_1_np, counts_2_np], **write_kwargs)

    dump = get_stage_debug_dump(renderer)
    da_tag = f"{data_access.name.lower()}_data_access" if data_access is not None else "default"
    api_tag = "async_api" if use_async_api else "sync_api"
    tag = f"cpu-{da_tag}-{api_tag}"
    (test_output_dir / f"test_bind_array_attribute_{tag}.usda").write_text(dump, encoding="utf-8")

    assert "[100, 200]" in dump, "Debug dump should contain updated values (prim 1)"
    assert "[300, 400, 500, 600]" in dump, "Debug dump should contain updated values (prim 2)"

    binding.unbind()

    print(f"[PASS] bind_array_attribute ({tag}) test passed")


@pytest.mark.usd_scene("cube.usda")
@pytest.mark.parametrize("device", [Device.CPU, Device.CUDA])
def test_map_attribute(device: Device, renderer, usd_scene, test_output_dir):
    """Test map/unmap workflow on CPU and CUDA.

    Maps an attribute, writes data via NumPy (CPU) or Warp (CUDA),
    unmaps to apply changes, and verifies via debug dump.
    """

    # Load cube.usda test scene
    usd_handle = renderer.add_usd(str(usd_scene))
    assert usd_handle is not None, f"USD handle should be valid after loading {usd_scene}"

    # Create source matrix: identity with [0][0] = 99.0
    source_np = np.eye(4, dtype=np.float64)
    source_np[0][0] = 99.0

    # [snippet:map-attribute-cpu]
    with renderer.map_attribute(
        prim_paths=["/World/Cube"],
        attribute_name="omni:xform",
        dtype=np.float64,
        shape=(4, 4),
        device=device,
        device_id=0,
    ) as mapping:
        assert mapping is not None, "Mapping should be valid"
        assert mapping.map_handle != 0, "Map handle should be non-zero"

        if device == Device.CPU:
            # CPU: Direct NumPy assignment (zero-copy write)
            dest_np = np.from_dlpack(mapping.tensor)
            assert dest_np.shape == (1, 4, 4), f"Expected shape (1, 4, 4), got {dest_np.shape}"
            dest_np[:] = source_np
        else:
            # CUDA: Use wp.copy for CPU->GPU transfer
            source_wp = wp.from_numpy(source_np, dtype=wp.mat44d)
            dest_wp = wp.from_dlpack(mapping.tensor)
            assert dest_wp.shape == (1, 4, 4), f"Expected shape (1, 4, 4), got {dest_wp.shape}"
            wp.copy(dest=dest_wp, src=source_wp)
            wp.synchronize_stream(dest_wp.device.stream)  # Ensure transfer completes before unmap
    # Auto-unmapped here
    # [/snippet:map-attribute-cpu]

    dump = get_stage_debug_dump(renderer)
    (test_output_dir / f"test_map_attribute_{device.name}.usda").write_text(dump, encoding="utf-8")

    assert "custom matrix4d omni:fabric:localMatrix" in dump, "Debug dump should contain matrix attribute"
    assert (
        "( (99, 0, 0, 0), (0, 1, 0, 0), (0, 0, 1, 0), (0, 0, 0, 1) )" in dump
    ), "Debug dump should contain expected matrix values"

    print(f"[PASS] map_attribute test passed on {device.name}: Matrix mapped, written via wp.copy, verified")


@pytest.mark.usd_scene("cube.usda")
@pytest.mark.parametrize("device", [Device.CPU, Device.CUDA])
def test_map_attribute_existing_only(device: Device, renderer, usd_scene, test_output_dir):
    """Test PrimMode.EXISTING_ONLY ignores missing prims.

    Maps two prims where one doesn't exist. With PrimMode.EXISTING_ONLY,
    the operation succeeds and only the existing prim is written.
    """

    usd_handle = renderer.add_usd(str(usd_scene))
    assert usd_handle is not None, f"USD handle should be valid after loading {usd_scene}"

    # Create two source matrices with different [0][0] values using NumPy directly
    matrix0_np = np.eye(4, dtype=np.float64)
    matrix0_np[0][0] = 99.0

    matrix1_np = np.eye(4, dtype=np.float64)
    matrix1_np[0][0] = 11.0  # This won't be used (prim missing)

    # Stack into (2, 4, 4) array
    source_np = np.stack([matrix0_np, matrix1_np], axis=0)

    with renderer.map_attribute(
        prim_paths=["/World/Cube", "/World/Cube1"],  # Cube1 doesn't exist
        attribute_name="omni:xform",
        dtype=np.float64,
        shape=(4, 4),
        prim_mode=PrimMode.EXISTING_ONLY,
        device=device,
        device_id=0,
    ) as mapping:
        assert mapping is not None, "Mapping should be valid"
        # Shape: (2, 4, 4) - two slots allocated even though second prim missing

        if device == Device.CPU:
            dest_np = np.from_dlpack(mapping.tensor)
            assert dest_np.shape == (2, 4, 4), f"Expected shape (2, 4, 4), got {dest_np.shape}"
            dest_np[:] = source_np
        else:
            source_wp = wp.from_numpy(source_np, dtype=wp.mat44d)
            dest_wp = wp.from_dlpack(mapping.tensor)
            assert dest_wp.shape == (2, 4, 4), f"Expected shape (2, 4, 4), got {dest_wp.shape}"
            wp.copy(dest=dest_wp, src=source_wp)
            wp.synchronize_stream(dest_wp.device.stream)
    # Auto-unmapped here

    dump = get_stage_debug_dump(renderer)
    (test_output_dir / f"test_map_attribute_existing_only_{device.name}.usda").write_text(dump, encoding="utf-8")

    assert "( (99, 0, 0, 0), (0, 1, 0, 0), (0, 0, 1, 0), (0, 0, 0, 1) )" in dump

    print(f"[PASS] map_attribute_existing_only test passed on {device.name}")


@pytest.mark.usd_scene("cube.usda")
def test_map_attribute_must_exist_failure(renderer, usd_scene, test_output_dir):
    """Test PrimMode.MUST_EXIST fails on missing prim.

    Maps a non-existing prim with PrimMode.MUST_EXIST.
    Map succeeds (deferred validation), but unmap should fail.
    """

    usd_handle = renderer.add_usd(str(usd_scene))
    assert usd_handle is not None, f"USD handle should be valid after loading {usd_scene}"

    mapping = renderer.map_attribute(
        prim_paths=["/World/Cube1"],  # Doesn't exist
        attribute_name="omni:xform",
        dtype=np.float64,
        shape=(4, 4),
        prim_mode=PrimMode.MUST_EXIST,
    )

    # Unmap should raise RuntimeError (works in both pytest and standalone execution)
    try:
        renderer.unmap_attribute(mapping)
        raise AssertionError("Expected RuntimeError but none was raised")
    except RuntimeError as e:
        # Verify error message indicates the prim was not found
        error_text = str(e).lower()
        assert "path or attribute not found" in error_text, f"Unexpected error message: {e}"

    print("[PASS] map_attribute_must_exist_failure test passed")


@pytest.mark.usd_scene("cube.usda")
def test_map_attribute_gpu(renderer, usd_scene, test_output_dir):
    """Test GPU map/unmap workflow with NumPy->GPU copy."""

    usd_handle = renderer.add_usd(str(usd_scene))
    assert usd_handle is not None, f"USD handle should be valid after loading {usd_scene}"

    # Build source matrices in NumPy: [0][0]=777,[3][3]=888 and [0][0]=555,[3][3]=666
    m0 = np.eye(4, dtype=np.float64)
    m0[0][0], m0[3][3] = 777.0, 888.0
    m1 = np.eye(4, dtype=np.float64)
    m1[0][0], m1[3][3] = 555.0, 666.0
    source_wp = wp.from_numpy(np.stack([m0, m1]), dtype=wp.mat44d)

    # [snippet:map-attribute-gpu]
    with renderer.map_attribute(
        prim_paths=["/World/Cube", "/World/Cube1"],
        attribute_name="omni:xform",
        semantic=Semantic.XFORM_MAT4x4,
        prim_mode=PrimMode.CREATE_NEW,
        device=Device.CUDA,
        device_id=0,
    ) as mapping:
        assert mapping.tensor.shape == (2, 4, 4), f"Expected (2, 4, 4), got {mapping.tensor.shape}"

        dest_wp = wp.from_dlpack(mapping.tensor, dtype=wp.mat44d)
        wp.copy(dest=dest_wp, src=source_wp)
        wp.synchronize_stream(dest_wp.device.stream)
    # Auto-unmapped here
    # [/snippet:map-attribute-gpu]

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

    usd_handle = renderer.add_usd(str(usd_scene))
    assert usd_handle is not None, f"USD handle should be valid after loading {usd_scene}"

    # Build source matrices in NumPy: [0][0]=777,[3][3]=888 and [0][0]=555,[3][3]=666
    m0 = np.eye(4, dtype=np.float64)
    m0[0][0], m0[3][3] = 777.0, 888.0
    m1 = np.eye(4, dtype=np.float64)
    m1[0][0], m1[3][3] = 555.0, 666.0
    source_wp = wp.from_numpy(np.stack([m0, m1]), dtype=wp.mat44d)

    # [snippet:unmap-attribute-cuda-sync]
    with renderer.map_attribute(
        prim_paths=["/World/Cube", "/World/Cube1"],
        attribute_name="omni:xform",
        semantic=Semantic.XFORM_MAT4x4,
        device=Device.CUDA,
        prim_mode=PrimMode.CREATE_NEW,
    ) as mapping:
        assert mapping.device == Device.CUDA, f"Expected device=Device.CUDA, got {mapping.device!r}"

        # Copy CPU->GPU on the device's default stream (no explicit sync - unmap handles it)
        dest_wp = wp.from_dlpack(mapping.tensor, dtype=wp.mat44d)
        wp_stream = dest_wp.device.stream
        wp.copy(dest=dest_wp, src=source_wp)

        # Unmap with CUDA sync - C library waits on event/stream before consuming data
        if sync_mode == "event":
            cuda_event = wp_stream.record_event()
            mapping.unmap(event=cuda_event.cuda_event)
        else:
            mapping.unmap(stream=wp_stream.cuda_stream)
    # __exit__ sees _unmapped=True -> no-op
    # [/snippet:unmap-attribute-cuda-sync]

    # Verify data was written correctly (proves sync worked)
    dump = get_stage_debug_dump(renderer)
    (test_output_dir / f"test_unmap_cuda_sync_{sync_mode}.usda").write_text(dump, encoding="utf-8")

    assert "( (777, 0, 0, 0), (0, 1, 0, 0), (0, 0, 1, 0), (0, 0, 0, 888) )" in dump
    assert "( (555, 0, 0, 0), (0, 1, 0, 0), (0, 0, 1, 0), (0, 0, 0, 666) )" in dump

    print(f"[PASS] unmap_attribute CUDA sync ({sync_mode}) test passed")


@pytest.mark.usd_scene("cube.usda")
def test_write_attribute_matrix4d_legacy(renderer, usd_scene, test_output_dir):
    """Legacy regression guard: write_attribute with Matrix4d.to_dltensor()."""

    from ovrtx.math import Matrix4d

    usd_handle = renderer.add_usd(str(usd_scene))
    assert usd_handle is not None, f"USD handle should be valid after loading {usd_scene}"

    matrix = Matrix4d()
    matrix.SetIdentity()
    matrix[0][0] = 42.0

    renderer.write_attribute(
        prim_paths=["/World/Cube"],
        attribute_name="omni:xform",
        tensor=matrix.to_dltensor(),
        semantic=Semantic.XFORM_MAT4x4,
    )

    dump = get_stage_debug_dump(renderer)
    (test_output_dir / "test_write_attribute_matrix4d_legacy.usda").write_text(dump, encoding="utf-8")

    assert "( (42, 0, 0, 0), (0, 1, 0, 0), (0, 0, 1, 0), (0, 0, 0, 1) )" in dump
    print("[PASS] write_attribute Matrix4d legacy test passed")


@pytest.mark.usd_scene("cube.usda")
def test_bind_write_explicit_semantic(renderer, usd_scene, test_output_dir):
    """Explicit semantic= path: bind + write with Semantic.XFORM_MAT4x4, verify round-trip."""

    usd_handle = renderer.add_usd(str(usd_scene))
    assert usd_handle is not None, f"USD handle should be valid after loading {usd_scene}"

    # [snippet:bind-write-semantic]
    binding = renderer.bind_attribute(
        prim_paths=["/World/Cube"],
        attribute_name="omni:xform",
        semantic=Semantic.XFORM_MAT4x4,
    )

    matrix = np.eye(4, dtype=np.float64)
    matrix[0][0] = 77.0
    binding.write(matrix.reshape(1, 4, 4))
    binding.unbind()
    # [/snippet:bind-write-semantic]

    dump = get_stage_debug_dump(renderer)
    (test_output_dir / "test_bind_write_explicit_semantic.usda").write_text(dump, encoding="utf-8")

    assert "( (77, 0, 0, 0), (0, 1, 0, 0), (0, 0, 1, 0), (0, 0, 0, 1) )" in dump
    print("[PASS] bind_write_explicit_semantic test passed")


@pytest.mark.usd_scene("cube.usda")
def test_write_attribute_token_string(renderer, usd_scene, test_output_dir):
    """Smoke test: write_attribute auto-detects list[str] as token strings."""

    usd_handle = renderer.add_usd(str(usd_scene))
    assert usd_handle is not None, f"USD handle should be valid after loading {usd_scene}"

    # [snippet:write-attribute-token-string]
    renderer.write_attribute(
        prim_paths=["/World/Cube"],
        attribute_name="testTokenAttr",
        tensor=["tokenValue1"],
        prim_mode=PrimMode.CREATE_NEW,
    )
    # [/snippet:write-attribute-token-string]

    dump = get_stage_debug_dump(renderer)
    (test_output_dir / "test_write_attribute_token_string.usda").write_text(dump, encoding="utf-8")

    assert '"tokenValue1"' in dump, "Token value should appear in debug dump"
    print("[PASS] write_attribute token_string test passed")


@pytest.mark.usd_scene("cube.usda")
@pytest.mark.parametrize(
    "source",
    [
        pytest.param("numpy", id="numpy"),
        pytest.param("warp", id="warp"),
    ],
)
def test_write_attribute_create_new_bool(renderer, usd_scene, test_output_dir, source):
    """Create a bool attribute via CREATE_NEW using different tensor backends."""

    usd_handle = renderer.add_usd(str(usd_scene))
    assert usd_handle is not None, f"USD handle should be valid after loading {usd_scene}"

    if source == "numpy":
        tensor = np.array([True], dtype=np.bool_)
    else:
        tensor = wp.array([True], dtype=wp.bool, device="cpu")

    renderer.write_attribute(
        prim_paths=["/World/Cube"],
        attribute_name="resetXformStack",
        tensor=tensor,
        prim_mode=PrimMode.CREATE_NEW,
    )

    dump = get_stage_debug_dump(renderer)
    (test_output_dir / f"test_write_attribute_create_new_bool_{source}.usda").write_text(dump, encoding="utf-8")

    assert "resetXformStack" in dump, "Bool attribute should appear in debug dump"
    print(f"[PASS] write_attribute bool CREATE_NEW ({source}) test passed")


@pytest.mark.usd_scene("cube.usda")
@pytest.mark.parametrize(
    "source",
    [
        pytest.param("builtin", id="builtin"),
        pytest.param("numpy", id="numpy"),
        pytest.param("warp", id="warp"),
    ],
)
def test_bind_attribute_create_new_bool(renderer, usd_scene, test_output_dir, source):
    """Bind a bool attribute via CREATE_NEW with different dtype sources, write through binding."""

    usd_handle = renderer.add_usd(str(usd_scene))
    assert usd_handle is not None, f"USD handle should be valid after loading {usd_scene}"

    if source == "warp":
        dtype = wp.bool
        tensor = wp.array([True], dtype=wp.bool, device="cpu")
    else:
        dtype = np.bool_ if source == "numpy" else bool
        tensor = np.array([True], dtype=dtype)

    binding = renderer.bind_attribute(
        prim_paths=["/World/Cube"],
        attribute_name="resetXformStack",
        dtype=dtype,
        prim_mode=PrimMode.CREATE_NEW,
    )

    binding.write(tensor)

    dump = get_stage_debug_dump(renderer)
    (test_output_dir / f"test_bind_attribute_create_new_bool_{source}.usda").write_text(dump, encoding="utf-8")

    assert "resetXformStack" in dump, "Bool attribute should appear in debug dump"
    print(f"[PASS] bind_attribute bool CREATE_NEW ({source}) test passed")


@pytest.mark.usd_scene("cube.usda")
def test_write_array_attribute_path_string(renderer, usd_scene, test_output_dir):
    """Smoke test: write_array_attribute auto-detects list[list[str]] as path strings (relationships)."""

    usd_handle = renderer.add_usd(str(usd_scene))
    assert usd_handle is not None, f"USD handle should be valid after loading {usd_scene}"

    renderer.clone_usd("/World/Cube", ["/World/Cube1"])

    # Write multi-target relationship: each prim gets 2 path targets pointing to prims that
    # exist in cube.usda (/World and /World/Cube). Diverges from C test's single-target case
    # to exercise variable-length per-prim tensors.
    # [snippet:write-array-attribute-path-string]
    renderer.write_array_attribute(
        prim_paths=["/World/Cube", "/World/Cube1"],
        attribute_name="testPathAttr",
        tensors=[["/World", "/World/Cube1"], ["/World", "/World/Cube"]],
        prim_mode=PrimMode.CREATE_NEW,
    )
    # [/snippet:write-array-attribute-path-string]

    dump = get_stage_debug_dump(renderer)
    (test_output_dir / "test_write_array_attribute_path_string.usda").write_text(dump, encoding="utf-8")

    # Extract each prim's block via regex (robust against formatting changes)
    import re

    cube_match = re.search(r'def Mesh "Cube"\s*\{(.+?)\n    \}', dump, re.DOTALL)
    cube1_match = re.search(r'def Mesh "Cube1"\s*\{(.+?)\n    \}', dump, re.DOTALL)
    assert cube_match, "Cube prim block should exist in dump"
    assert cube1_match, "Cube1 prim block should exist in dump"

    cube_block = cube_match.group(1)
    cube1_block = cube1_match.group(1)

    assert "testPathAttr" in cube_block, "Cube should have testPathAttr"
    assert "</World>" in cube_block, "Cube should reference /World"
    assert "</World/Cube1>" in cube_block, "Cube should reference /World/Cube1"

    assert "testPathAttr" in cube1_block, "Cube1 should have testPathAttr"
    assert "</World>" in cube1_block, "Cube1 should reference /World"
    assert "</World/Cube>" in cube1_block, "Cube1 should reference /World/Cube"
    print("[PASS] write_array_attribute path_string test passed")


@pytest.mark.usd_scene("cube.usda")
def test_write_array_attribute_multi_component(renderer, usd_scene, test_output_dir):
    """Write multi-component point3f array to an existing mesh attribute.

    Verifies that (K, 3) float32 input is inferred as 3-component elements automatically.
    """

    usd_handle = renderer.add_usd(str(usd_scene))
    assert usd_handle is not None, f"USD handle should be valid after loading {usd_scene}"

    # [snippet:write-array-attribute-points]
    points = np.array([[111.0, 222.0, 333.0], [444.0, 555.0, 666.0]], dtype=np.float32)
    renderer.write_array_attribute(
        prim_paths=["/World/Cube"],
        attribute_name="points",
        tensors=[points],
    )
    # [/snippet:write-array-attribute-points]

    dump = get_stage_debug_dump(renderer)
    (test_output_dir / "test_write_array_attribute_multi_component.usda").write_text(dump, encoding="utf-8")

    assert "(111, 222, 333)" in dump, "First point should appear in dump"
    assert "(444, 555, 666)" in dump, "Second point should appear in dump"
    print("[PASS] write_array_attribute multi-component (point3f inference) test passed")


def test_write_attribute_path_string_rejects_scalar(renderer):
    """Python-only guard: path_string semantic must use write_array_attribute, not write_attribute."""

    with pytest.raises(ValueError, match="PATH_STRING.*write_array_attribute"):
        renderer.write_attribute(
            prim_paths=["/World/Cube"],
            attribute_name="testPathAttr",
            tensor=["/World/Cube1"],
            semantic=Semantic.PATH_STRING,
        )

    print("[PASS] path_string scalar rejection test passed")


def test_write_attribute_token_string_type_validation(renderer):
    """Python-only guard: string semantic with non-List[str] input raises TypeError."""

    with pytest.raises(TypeError, match="expected a list of strings"):
        renderer.write_attribute(
            prim_paths=["/World/Cube"],
            attribute_name="testTokenAttr",
            tensor=np.array([1, 2], dtype=np.int32),
            semantic=Semantic.TOKEN_STRING,
        )

    print("[PASS] token_string type validation test passed")


def test_write_attribute_async_string_rejects_async(renderer):
    """Python-side guard: DataAccess.ASYNC with string attributes raises ValueError."""

    with pytest.raises(ValueError, match="String attributes.*DataAccess.SYNC"):
        renderer.write_attribute_async(
            prim_paths=["/World/Cube"],
            attribute_name="testTokenAttr",
            tensor=["tokenValue1"],
            semantic=Semantic.TOKEN_STRING,
            data_access=DataAccess.ASYNC,
        )

    print("[PASS] string + async rejection test passed")


@pytest.mark.usd_scene("simple_scene.usda")
def test_renderer(renderer, usd_scene, test_output_dir):
    """Test basic renderer functionality with multiple render products."""

    from PIL import Image

    # Note: Renderer creation/deletion is tested implicitly by session fixture lifecycle.
    # The fixture calls reset_stage() before each test, ensuring clean state.

    print(f"Successfully created renderer with config:\n{renderer.config}")

    # Load USD file
    usd_handle = renderer.add_usd(str(usd_scene))
    assert usd_handle is not None, f"USD handle should be valid after loading {usd_scene}"
    print(f"Successfully loaded USD file, handle: {usd_handle}")

    render_products = {
        "/Render/OmniverseKit/HydraTextures/ViewportTexture0",
        "/Render/OmniverseKit/HydraTextures/ViewportTexture1",
        "/Render/OmniverseKit/HydraTextures/ViewportTexture2",
    }

    # Warm up the renderer (path tracer needs a few frames to converge)
    for i in range(10):
        products = renderer.step(render_products=render_products, delta_time=0.1)
        assert products is not None, f"Step products should be valid at warm-up step {i}"
        frame_counts = ", ".join(f"{name.split('/')[-1]}={len(p.frames)}" for name, p in products.items())
        print(f"  warm-up step elapsed time: {(i + 1) * 0.1:.1f}s - number of frames: {frame_counts}")

    # Render final frame — map, apply Sobel, and save LdrColor outputs
    # [snippet:step-and-read-output]
    products = renderer.step(render_products=render_products, delta_time=0.1)
    assert products is not None, "Final step products should be valid"

    saved_files = []
    for product_name, product in products.items():
        print(f"  Saving final frames for product: {product_name}")

        for frame in product.frames:
            assert "LdrColor" in frame.render_vars, "LdrColor render variable should be present"

            # Verify frame timing is in expected range (10 warm-up + 1 final = 11 steps × 0.1s = 1.1s)
            assert frame.end_time >= 1.0, f"Final frame end_time {frame.end_time:.2f}s should be >= 1.0s"

            # Save source LdrColor as PNG
            filename_prefix = f"{product_name.split('/')[-1]}_LdrColor"
            render_var = frame.render_vars["LdrColor"]
            with render_var.map(device=Device.CPU) as mapping:
                np_array = np.from_dlpack(mapping.tensor)
                src_file = test_output_dir / f"{filename_prefix}.png"
                Image.fromarray(np_array).save(src_file)
                saved_files.append(src_file)
            # [/snippet:step-and-read-output]

            for map_device in [Device.CPU, Device.CUDA]:
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
                    sobel_file = test_output_dir / f"{filename_prefix}_sobel_{map_device.name.lower()}.png"
                    Image.fromarray(wp_output.numpy()).save(sobel_file)
                    saved_files.append(sobel_file)

    # Verify all expected output files were written (1 source + 2 sobel per product)
    expected_files = len(render_products) * 3
    assert len(saved_files) == expected_files, f"Expected {expected_files} output files, got {len(saved_files)}"
    for f in saved_files:
        assert f.exists(), f"Output file should exist: {f}"
        assert f.stat().st_size > 0, f"Output file should not be empty: {f}"

    print("[PASS] renderer test passed")


@pytest.mark.usd_scene("simple_scene.usda")
@pytest.mark.parametrize("sync_mode", ["global", "event", "stream"])
def test_renderer_cuda(sync_mode: str, renderer, usd_scene, test_output_dir):
    """Test CUDA device mapping with different sync modes.

    - global: wp.synchronize_device() (baseline)
    - event: explicit CUDA event sync on unmap (custom stream)
    - stream: explicit CUDA stream sync on unmap (custom stream)

    Performs actual GPU work (copy) and verifies data integrity to prove
    synchronization worked correctly.
    """

    usd_handle = renderer.add_usd(str(usd_scene))
    assert usd_handle is not None, f"USD handle should be valid after loading {usd_scene}"

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
                with render_var.map(device=Device.CUDA) as mapping:
                    wp_image = wp.from_dlpack(mapping.tensor)

                    # Validate CUDA device
                    assert wp_image.device.is_cuda, f"Expected CUDA device, got {wp_image.device}"
                    device_ids.add(str(wp_image.device))
                    total_outputs += 1

                    np_image = wp_image.numpy()  # obtain an early CPU copy of the source data before cloning on GPU

                    # Copy data over zeroed destination on the device's stream
                    wp_stream = wp_image.device.stream
                    dest = wp.zeros_like(wp_image, device=wp_image.device)
                    wp.copy(dest, wp_image)

                    # Exercise different explicit ovrtx sync modes
                    if sync_mode == "event":
                        cuda_event = wp_stream.record_event()
                        mapping.unmap(event=cuda_event.cuda_event)
                    elif sync_mode == "stream":
                        mapping.unmap(stream=wp_stream.cuda_stream)
                    else:  # "global"
                        wp.synchronize_device(wp_image.device)

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

    # Define Warp kernel for computing sum over flat uint8 buffer
    @wp.kernel
    def compute_sum_kernel(data: wp.array(dtype=wp.uint8), result: wp.array(dtype=wp.float64)):
        i = wp.tid()
        pixel_value = wp.float64(data[i])  # Explicit float64 conversion
        wp.atomic_add(result, 0, pixel_value)

    def compute_sum_on_device(render_var, device: Device, var_name: str) -> float:
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
                f"  {device.name:4s}: {var_name} shape={wp_array.shape} dtype={wp_array.dtype} "
                f"device={wp_array.device} contiguous={wp_array.is_contiguous} sum={computed_sum:.1f}"
            )

            return computed_sum

    usd_handle = renderer.add_usd(str(usd_scene))
    assert usd_handle is not None, f"USD handle should be valid after loading {usd_scene}"

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

                cuda_sum = compute_sum_on_device(render_var, Device.CUDA, var_name)
                cpu_sum = compute_sum_on_device(render_var, Device.CPU, var_name)

                # Validate consistency with float tolerance
                sum_diff = abs(cuda_sum - cpu_sum)
                print(f"  Consistency: CUDA vs CPU sum diff = {sum_diff:.1f}")
                assert sum_diff < 1e-3, f"CUDA and CPU saw different data! Diff:{sum_diff:.1f}"

    print("[PASS] renderer Warp interop test passed")


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

    usd_handle = renderer.add_usd(str(usd_scene))
    assert usd_handle is not None, f"USD handle should be valid after loading {usd_scene}"

    render_products = {"/Render/OmniverseKit/HydraTextures/ViewportTexture0"}
    # [snippet:step-async-polling]
    result = renderer.step_async(render_products=render_products, delta_time=1 / 60)

    # Phase 1: Immediate poll - should timeout (non-blocking)
    metadata = result.wait(step_timeout_ns=0)
    assert metadata is None, "Expected timeout on immediate poll"
    assert not result.step_complete, "step_complete must be False after timeout"

    # Phase 2: Poll again with sufficient timeout - should succeed
    metadata = result.wait(step_timeout_ns=10_000_000_000)  # 10 sec
    # [/snippet:step-async-polling]
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


@pytest.mark.xfail(reason="Multi-GPU stability investigation pending", strict=False)
@pytest.mark.usd_scene("simple_scene.usda")
@pytest.mark.skipif(wp.get_cuda_device_count() < 2, reason="Requires 2+ CUDA devices")
@pytest.mark.parametrize("src_device_id", [0, 1])
def test_mgpu_gpu_transforms(src_device_id, usd_scene, test_output_dir):
    """Test GPU transform propagation across multiple GPUs.

    Mirrors the C test "OVRTX - Multi-GPU GPU Transforms": creates a dedicated renderer
    with read_gpu_transforms enabled, writes transforms from a specific CUDA device, and
    verifies the transform effect is visible in all render products (which may execute on
    different GPUs).

    Uses before/after comparison: captures a baseline frame with the cube at its original
    position, then writes a large translation from the source GPU to move the cube off-screen.
    Both render products must show a significant image change (normalized RMSE > 1%), proving
    the transform propagated to all rendering GPUs.

    Threshold justification:
    - The cube covers ~5-15% of each camera's view with distinct blue material.
    - Moving it off-screen changes those pixels by ~50-200 intensity per channel.
    - Expected normalized RMSE from cube removal: ~0.02-0.10.
    - Denoiser frame-to-frame noise for a converged scene: < 0.005 normalized RMSE.
    - Threshold at 0.01 sits safely between noise floor and expected signal.
    """

    from PIL import Image

    WARM_UP_FRAMES = 10  # match C test frame count
    DELTA_TIME = 1.0 / 30.0
    RMSE_THRESHOLD = 0.01

    # ovrtx (via carb.cudainterop) uses cudaMallocAsync which allocates from the default
    # CUDA memory pool. Cross-device cudaMemcpyPeerAsync requires explicit mempool access
    # grants — normally done by the Hydra GpuInterop component, but not loaded in standalone
    # ovrtx. Grant bidirectional access so mempool pointers are visible across devices.
    for i in range(wp.get_cuda_device_count()):
        for j in range(wp.get_cuda_device_count()):
            if i != j:
                wp.set_mempool_access_enabled(f"cuda:{i}", f"cuda:{j}", True)

    config = RendererConfig(
        read_gpu_transforms=True,
        log_file_path=str(test_output_dir / f"test_mgpu_gpu_transforms_src{src_device_id}.log"),
        log_level="info",
    )
    mgpu_renderer = Renderer(config)
    try:
        usd_handle = mgpu_renderer.add_usd(str(usd_scene))
        assert usd_handle is not None, f"USD handle should be valid after loading {usd_scene}"

        render_products = {
            "/Render/OmniverseKit/HydraTextures/ViewportTexture0",
            "/Render/OmniverseKit/HydraTextures/ViewportTexture1",
        }

        for i in range(WARM_UP_FRAMES):
            products = mgpu_renderer.step(render_products=render_products, delta_time=DELTA_TIME)
            assert products is not None, f"Step should succeed at warm-up frame {i}"

        baseline_products = mgpu_renderer.step(render_products=render_products, delta_time=DELTA_TIME)
        baseline_images = {}
        for product_name, product in baseline_products.items():
            for frame in product.frames:
                if "LdrColor" in frame.render_vars:
                    with frame.render_vars["LdrColor"].map(device=Device.CPU) as mapping:
                        baseline_images[product_name] = np.from_dlpack(mapping.tensor).copy()

        assert len(baseline_images) == len(
            render_products
        ), f"Expected baseline images for all {len(render_products)} products, got {len(baseline_images)}"

        transform_np = np.eye(4, dtype=np.float64)
        transform_np[3, 0] = 99999.0
        transform_np = transform_np.reshape(1, 4, 4)

        wp_device = f"cuda:{src_device_id}"
        tensor = wp.from_numpy(transform_np, dtype=wp.mat44d, device=wp_device)
        wp_stream = tensor.device.stream

        mgpu_renderer.write_attribute(
            prim_paths=["/World/Cube"],
            attribute_name="omni:xform",
            tensor=tensor,
            data_access=DataAccess.ASYNC,
            cuda_stream=wp_stream.cuda_stream,
        )

        result_products = mgpu_renderer.step(render_products=render_products, delta_time=DELTA_TIME)

        verified_products = set()
        for product_name, product in result_products.items():
            for frame in product.frames:
                if "LdrColor" not in frame.render_vars:
                    continue
                with frame.render_vars["LdrColor"].map(device=Device.CPU) as mapping:
                    result_image = np.from_dlpack(mapping.tensor).copy()

                baseline = baseline_images[product_name]
                diff = result_image.astype(np.float32) - baseline.astype(np.float32)
                rmse = float(np.sqrt(np.mean(diff**2)) / 255.0)

                short_name = product_name.split("/")[-1]
                print(f"  {short_name}: normalized RMSE = {rmse:.4f} (threshold: {RMSE_THRESHOLD})")

                Image.fromarray(baseline).save(test_output_dir / f"mgpu_src{src_device_id}_{short_name}_baseline.png")
                Image.fromarray(result_image).save(test_output_dir / f"mgpu_src{src_device_id}_{short_name}_result.png")

                assert rmse > RMSE_THRESHOLD, (
                    f"Transform from cuda:{src_device_id} did not visibly affect {short_name}: "
                    f"RMSE={rmse:.4f} <= {RMSE_THRESHOLD} — cross-GPU propagation may have failed"
                )
                verified_products.add(product_name)

        assert verified_products == render_products, (
            f"Not all render products produced valid LdrColor output. "
            f"Verified: {sorted(p.split('/')[-1] for p in verified_products)}, "
            f"expected: {sorted(p.split('/')[-1] for p in render_products)}"
        )

        print(f"[PASS] mgpu_gpu_transforms (src_device={src_device_id})")

    finally:
        del mgpu_renderer


def test_raw_library(test_output_dir):
    """Test raw library bindings directly (no fixtures, does its own init/shutdown).

    This test must run last as it manipulates library ref-counting directly.
    """

    import ctypes

    from ovrtx._src.bindings import OVRTX_API_SUCCESS, _ovrtx_loader

    _OVRTX_TEST_CONFIG = RendererConfig(
        # Configure logging via OVRTX config
        log_level="info",
        log_file_path=test_output_dir,
    )
    ovrtx_config = Renderer._to_c_config(_OVRTX_TEST_CONFIG)
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

    print("[PASS] raw library bindings test passed")

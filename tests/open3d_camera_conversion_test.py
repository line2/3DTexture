#!/usr/bin/env python3
"""Verify the orthographic-to-virtual-pinhole camera conversion."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import open3d as o3d


def load_module():
    path = Path(__file__).resolve().parents[1] / "tools" / "open3d_project_to_albedo.py"
    spec = importlib.util.spec_from_file_location("open3d_project_to_albedo", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_exact_vertex_welding(module) -> None:
    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.0, 1.0, 0.0],
            [2.0, 0.0, 0.0],
        ]
    )
    mesh.triangles = o3d.utility.Vector3iVector(
        [[0, 1, 2], [2, 3, 4], [0, 1, 5]]
    )
    mesh.triangle_uvs = o3d.utility.Vector2dVector(
        np.arange(18, dtype=np.float64).reshape((-1, 2))
    )

    info = module.weld_duplicate_vertices(mesh)
    assert len(mesh.vertices) == 5
    assert len(mesh.triangles) == 2
    assert len(mesh.triangle_uvs) == 6
    assert info["removed_duplicate_vertices"] == 1
    assert info["removed_degenerate_triangles"] == 1


def main() -> None:
    module = load_module()
    test_exact_vertex_welding(module)
    image = o3d.t.geometry.Image(np.zeros((8, 8, 3), dtype=np.uint8))

    # Columns are image-right, image-up, and camera-backward.  This camera looks
    # along +Y from y=-2.2 and uses +Z as image-up.
    camera_to_world = np.array(
        [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, -1.0, -2.2],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )
    view = {
        "camera": {
            "camera_to_world": camera_to_world.tolist(),
            "render_ortho_scale": 1.0,
        },
        "spec": {},
    }

    intrinsic, extrinsic = module.make_virtual_pinhole(view, image, 1000.0)
    right = camera_to_world[:3, 0]
    up = camera_to_world[:3, 1]
    backward = camera_to_world[:3, 2]
    rotation = extrinsic[:3, :3]
    assert np.allclose(rotation[0], right)
    assert np.allclose(rotation[1], -up)
    assert np.allclose(rotation[2], -backward)

    point = np.array([0.2, 0.3, 0.4])
    pixel_scale = 8.0 / 1.0
    exact = np.array(
        [
            pixel_scale * point.dot(right) + 4.0,
            pixel_scale * point.dot(-up) + 4.0,
        ]
    )
    homogeneous = extrinsic @ np.append(point, 1.0)
    projected = intrinsic @ homogeneous[:3]
    virtual = projected[:2] / projected[2]
    assert np.allclose(virtual, exact, atol=0.01), (virtual, exact)

    print("Open3D camera conversion test passed")


if __name__ == "__main__":
    main()

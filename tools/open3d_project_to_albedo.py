#!/usr/bin/env python3
"""Project calibrated orthographic views onto an existing UV atlas.

This is a Python/Open3D implementation modeled on Open3D's
``triangle_mesh_project_to_albedo.py`` example.  The environment is managed by
uv through ``pyproject.toml`` and ``uv.lock``.

For the ``test_04`` metadata format, cameras are stored as camera-to-world
matrices.  Open3D's ``project_images_to_albedo`` API accepts world-to-camera
pinhole cameras, so this tool:

1. welds exact duplicate XYZ vertices before any projection work,
2. converts OBJ coordinates to the render coordinate system when needed,
3. estimates/uses the uniform mesh scale used by the renderer,
4. converts each orthographic camera into an equivalent pinhole camera at a
   large virtual distance, and
5. calls ``TriangleMesh.project_images_to_albedo``.

The virtual pinhole conversion is intentionally explicit.  It is very close to
orthographic for normalized meshes, but it is still a finite-perspective
approximation because Open3D 0.19 has no orthographic overload.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import open3d as o3d
from open3d.core import Tensor


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "dataset",
        type=Path,
        help="Dataset directory containing eval_metadata.json and view images",
    )
    parser.add_argument(
        "--mesh",
        type=Path,
        help="Mesh with an existing UV atlas (default: the sole OBJ in dataset)",
    )
    parser.add_argument(
        "--metadata",
        type=Path,
        help="Metadata JSON (default: DATASET/eval_metadata.json)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory (default: DATASET/open3d_out)",
    )
    parser.add_argument(
        "--tex-size",
        type=int,
        default=2048,
        help="Square output texture size (default: 2048)",
    )
    parser.add_argument(
        "--coordinate-frame",
        choices=("auto", "obj", "render"),
        default="auto",
        help=(
            "Input mesh coordinate system. 'obj' applies (x,y,z)->(x,-z,y). "
            "'render' leaves XYZ unchanged. 'auto' compares both choices "
            "with camera_fit metadata."
        ),
    )
    parser.add_argument(
        "--mesh-scale",
        type=float,
        default=None,
        help=(
            "Uniform mesh multiplier applied after coordinate conversion. "
            "By default it is estimated from camera_fit metadata."
        ),
    )
    parser.add_argument(
        "--virtual-distance",
        type=float,
        default=0.0,
        help=(
            "Virtual pinhole camera distance. 0 selects a conservative large "
            "distance based on mesh radius and original camera depth "
            "(default: 0). Larger values better approximate strict "
            "orthographic projection."
        ),
    )
    parser.add_argument(
        "--contrast",
        type=float,
        default=1.0,
        help="Optional albedo linear contrast multiplier (default: 1.0)",
    )
    parser.add_argument(
        "--views",
        default=None,
        help="Comma-separated selected view names (default: metadata order)",
    )
    parser.add_argument(
        "--no-weld",
        action="store_true",
        help=(
            "Disable exact vertex welding. By default, vertices with identical "
            "XYZ coordinates are merged before projection."
        ),
    )
    parser.add_argument(
        "--no-glb",
        action="store_true",
        help="Write albedo.png and calibration data, but skip mesh_albedo.glb",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow writing into a non-empty output directory",
    )
    args = parser.parse_args()

    if args.tex_size < 1 or args.tex_size > 16384:
        parser.error("--tex-size must be in [1, 16384]")
    if args.virtual_distance < 0:
        parser.error("--virtual-distance must be non-negative")
    if args.contrast <= 0:
        parser.error("--contrast must be positive")
    if args.mesh_scale is not None and args.mesh_scale <= 0:
        parser.error("--mesh-scale must be positive")
    return args


def find_mesh(dataset: Path, explicit: Path | None) -> Path:
    if explicit is not None:
        mesh = explicit if explicit.is_absolute() else dataset / explicit
        if not mesh.is_file():
            raise FileNotFoundError(f"Mesh not found: {mesh}")
        return mesh.resolve()

    excluded = {"result.obj", "label.obj"}
    candidates = [
        path
        for path in sorted(dataset.glob("*.obj"))
        if path.name.lower() not in excluded
    ]
    uv_candidates = [path for path in candidates if "_uv" in path.stem.lower()]
    if len(uv_candidates) == 1:
        return uv_candidates[0].resolve()
    if len(candidates) == 1:
        return candidates[0].resolve()
    if not candidates:
        raise FileNotFoundError(f"No OBJ mesh found in {dataset}")
    names = ", ".join(path.name for path in candidates)
    raise RuntimeError(f"Multiple OBJ meshes found; select one with --mesh: {names}")


def load_metadata(dataset: Path, explicit: Path | None) -> dict[str, Any]:
    metadata_path = explicit or dataset / "eval_metadata.json"
    if not metadata_path.is_file():
        raise FileNotFoundError(f"Metadata not found: {metadata_path}")
    with metadata_path.open("r", encoding="utf-8") as stream:
        return json.load(stream)


def collect_manifest_views(metadata: dict[str, Any]) -> dict[str, dict[str, Any]]:
    views: dict[str, dict[str, Any]] = {}
    for shard in metadata.get("shards", []):
        manifest = shard.get("global_control_render_manifest", {})
        for view in manifest.get("views", []):
            label = view.get("label") or view.get("name")
            if label:
                views[str(label)] = view
    return views


def selected_view_specs(metadata: dict[str, Any]) -> list[dict[str, Any]]:
    views = list(metadata.get("selected_views", []))
    if not views:
        specs = metadata.get("selection_meta", {}).get("selected_specs", [])
        views = specs
    if not views:
        raise ValueError("Metadata has no selected_views")
    views.sort(key=lambda view: int(view.get("selected_index", 0)))
    return views


def image_path_for_view(dataset: Path, view: dict[str, Any]) -> Path:
    label = str(view.get("label") or view.get("name"))
    candidates = [
        dataset / f"{label}.png",
        dataset / f"{label}.jpg",
        dataset / f"{label}.jpeg",
        dataset / f"{label}.PNG",
        dataset / f"{label}.JPG",
    ]
    for path in candidates:
        if path.is_file():
            return path.resolve()

    explicit = view.get("image_path")
    if explicit:
        path = Path(explicit)
        if path.is_file():
            return path.resolve()
        local = dataset / path.name
        if local.is_file():
            return local.resolve()
    raise FileNotFoundError(f"No image found for view {label!r} in {dataset}")


def load_views(
    dataset: Path, metadata: dict[str, Any], selected_names: list[str] | None
) -> list[dict[str, Any]]:
    manifest_views = collect_manifest_views(metadata)
    specs = selected_view_specs(metadata)
    if selected_names:
        by_name = {str(view.get("name")): view for view in specs}
        missing = [name for name in selected_names if name not in by_name]
        if missing:
            raise ValueError(f"Views not selected in metadata: {', '.join(missing)}")
        specs = [by_name[name] for name in selected_names]

    loaded: list[dict[str, Any]] = []
    for spec in specs:
        name = str(spec.get("name"))
        if name not in manifest_views:
            raise KeyError(f"Camera manifest is missing selected view {name!r}")
        camera = manifest_views[name]
        if str(camera.get("projection_type", "")).upper() != "ORTHO":
            raise ValueError(
                f"View {name!r} is {camera.get('projection_type')!r}; "
                "this adapter currently supports ORTHO metadata"
            )
        if "camera_to_world" not in camera:
            raise KeyError(f"View {name!r} has no camera_to_world matrix")
        scale = camera.get("render_ortho_scale")
        if scale is None or float(scale) <= 0:
            raise ValueError(f"View {name!r} has invalid render_ortho_scale")
        image = image_path_for_view(dataset, camera)
        loaded.append(
            {
                "name": name,
                "spec": spec,
                "camera": camera,
                "image": image,
            }
        )
    return loaded


def obj_to_render(vertices: np.ndarray) -> np.ndarray:
    result = vertices[:, [0, 2, 1]].copy()
    result[:, 1] *= -1.0
    return result


def projection_extents(
    render_vertices: np.ndarray, camera_to_world: np.ndarray
) -> tuple[float, float]:
    c2w = np.asarray(camera_to_world, dtype=np.float64)
    right = c2w[:3, 0]
    up = c2w[:3, 1]
    x = render_vertices @ right
    y = render_vertices @ -up
    return float(np.max(x) - np.min(x)), float(np.max(y) - np.min(y))


def fit_records(metadata: dict[str, Any]) -> list[dict[str, Any]]:
    for shard in metadata.get("shards", []):
        manifest = shard.get("global_control_render_manifest", {})
        fit = manifest.get("camera_fit", {}).get("selected_fit", {})
        records = fit.get("per_view", [])
        if records:
            return records
    return []


def estimate_scale_for_transform(
    vertices: np.ndarray,
    metadata: dict[str, Any],
    views: list[dict[str, Any]],
    transform_obj: bool,
) -> tuple[float, float]:
    records = {str(record.get("name")): record for record in fit_records(metadata)}
    ratios: list[float] = []
    measurements: list[tuple[float, float]] = []
    for view in views:
        name = view["name"]
        record = records.get(name)
        if record is None:
            continue
        camera = np.asarray(view["camera"]["camera_to_world"], dtype=np.float64)
        transformed = obj_to_render(vertices) if transform_obj else vertices
        x_extent, y_extent = projection_extents(transformed, camera)
        fit_x = float(record["x_max"]) - float(record["x_min"])
        fit_y = float(record["y_max"]) - float(record["y_min"])
        for measured, expected in ((x_extent, fit_x), (y_extent, fit_y)):
            if measured > 1e-12 and expected > 1e-12:
                ratios.append(expected / measured)
                measurements.append((measured, expected))
    if not ratios:
        return 1.0, float("inf")
    scale = float(np.median(ratios))
    error = float(
        np.median(
            [
                abs(measured * scale - expected) / expected
                for measured, expected in measurements
            ]
        )
    )
    if not np.isfinite(scale) or scale <= 0:
        return 1.0, float("inf")
    return scale, error


def weld_duplicate_vertices(mesh: o3d.geometry.TriangleMesh) -> dict[str, Any]:
    """Merge exact XYZ duplicates and remove resulting degenerate faces.

    Open3D stores OBJ UVs per triangle corner, so merging vertex indices does
    not discard UV seams. This is deliberately an exact-coordinate weld; it
    does not move geometry or merge points that are merely close.
    """
    input_vertices = len(mesh.vertices)
    input_triangles = len(mesh.triangles)
    if input_vertices == 0 or input_triangles == 0:
        raise ValueError("Cannot weld an empty mesh")

    mesh.remove_duplicated_vertices()

    triangles = np.asarray(mesh.triangles)
    vertices = np.asarray(mesh.vertices)
    removed_degenerate_triangles = 0
    if triangles.shape[0] != 0:
        edge_1 = vertices[triangles[:, 1]] - vertices[triangles[:, 0]]
        edge_2 = vertices[triangles[:, 2]] - vertices[triangles[:, 0]]
        area_2 = np.linalg.norm(np.cross(edge_1, edge_2), axis=1)
        keep_face = area_2 > 0.0
        removed_degenerate_triangles = int(np.count_nonzero(~keep_face))
        if removed_degenerate_triangles:
            triangle_uvs = np.asarray(mesh.triangle_uvs)
            if triangle_uvs.shape[0] != 3 * input_triangles:
                raise ValueError("Unexpected triangle UV count while welding")
            mesh.triangles = o3d.utility.Vector3iVector(
                triangles[keep_face].astype(np.int32)
            )
            mesh.triangle_uvs = o3d.utility.Vector2dVector(
                triangle_uvs.reshape((-1, 3, 2))[keep_face].reshape((-1, 2))
            )

    if len(mesh.triangles) == 0:
        raise ValueError("Vertex welding removed all triangles")

    return {
        "enabled": True,
        "method": "exact_xyz",
        "input_vertices": input_vertices,
        "output_vertices": len(mesh.vertices),
        "removed_duplicate_vertices": input_vertices - len(mesh.vertices),
        "input_triangles": input_triangles,
        "output_triangles": len(mesh.triangles),
        "removed_degenerate_triangles": removed_degenerate_triangles,
    }


def prepare_mesh(
    mesh_path: Path,
    metadata: dict[str, Any],
    views: list[dict[str, Any]],
    coordinate_frame: str,
    mesh_scale: float | None,
    weld: bool,
) -> tuple[o3d.geometry.TriangleMesh, dict[str, Any]]:
    legacy = o3d.io.read_triangle_mesh(str(mesh_path))
    if len(legacy.vertices) == 0 or len(legacy.triangles) == 0:
        raise ValueError(f"Mesh is empty: {mesh_path}")
    if len(legacy.triangle_uvs) != 3 * len(legacy.triangles):
        raise ValueError(
            "Mesh has no per-corner UV coordinates. Open3D's "
            "project_images_to_albedo requires an existing UV atlas."
        )

    welding = (
        weld_duplicate_vertices(legacy)
        if weld
        else {
            "enabled": False,
            "method": "disabled",
            "input_vertices": len(legacy.vertices),
            "output_vertices": len(legacy.vertices),
            "removed_duplicate_vertices": 0,
            "input_triangles": len(legacy.triangles),
            "output_triangles": len(legacy.triangles),
            "removed_degenerate_triangles": 0,
        }
    )

    vertices = np.asarray(legacy.vertices, dtype=np.float64)
    obj_scale, obj_error = estimate_scale_for_transform(vertices, metadata, views, True)
    render_scale, render_error = estimate_scale_for_transform(
        vertices, metadata, views, False
    )

    if coordinate_frame == "auto":
        has_fit = np.isfinite(obj_error) and np.isfinite(render_error)
        if has_fit and render_error <= obj_error:
            frame = "render"
            scale = render_scale
        else:
            frame = "obj"
            scale = obj_scale
    else:
        frame = coordinate_frame
        scale = obj_scale if frame == "obj" else render_scale

    if mesh_scale is not None:
        scale = mesh_scale

    if frame == "obj":
        transformed = obj_to_render(vertices) * scale
    else:
        transformed = vertices * scale
    legacy.vertices = o3d.utility.Vector3dVector(transformed)

    center = np.asarray(legacy.vertices).mean(axis=0)
    radius = float(np.linalg.norm(np.asarray(legacy.vertices) - center, axis=1).max())
    info = {
        "coordinate_frame": frame,
        "mesh_scale": scale,
        "mesh_center": center.tolist(),
        "mesh_radius": radius,
        "auto_fit_error_obj": obj_error,
        "auto_fit_error_render": render_error,
        "welding": welding,
    }
    return legacy, info


def max_original_camera_depth(views: list[dict[str, Any]]) -> float:
    depths: list[float] = []
    for view in views:
        c2w = np.asarray(view["camera"]["camera_to_world"], dtype=np.float64)
        depths.append(float(np.dot(c2w[:3, 3], c2w[:3, 2])))
    return max(depths, default=0.0)


def make_virtual_pinhole(
    view: dict[str, Any],
    image: o3d.t.geometry.Image,
    virtual_distance: float,
) -> tuple[np.ndarray, np.ndarray]:
    c2w = np.asarray(view["camera"]["camera_to_world"], dtype=np.float64)
    if c2w.shape != (4, 4):
        raise ValueError(f"camera_to_world must be 4x4, got {c2w.shape}")

    right = c2w[:3, 0]
    up = c2w[:3, 1]
    backward = c2w[:3, 2]
    original_center = c2w[:3, 3]
    original_depth = float(np.dot(original_center, backward))
    if virtual_distance <= original_depth + 1e-9:
        raise ValueError(
            "--virtual-distance must be greater than the original camera "
            f"distance ({original_depth:.9g})"
        )

    # Preserve the original camera's image-plane origin while moving the
    # camera to a large virtual depth.  This keeps the orthographic u/v offsets
    # unchanged instead of recentering the camera on the mesh.
    camera_center = original_center + backward * (virtual_distance - original_depth)

    # Open3D expects a world-to-camera matrix.  The metadata columns are
    # image-right, image-up, and camera-backward.  Open3D's camera z row points
    # forward, so rows 1 and 2 are image-down (-up) and camera-forward
    # (-backward).
    world_to_camera = np.vstack((right, -up, -backward))
    extrinsic = np.eye(4, dtype=np.float64)
    extrinsic[:3, :3] = world_to_camera
    extrinsic[:3, 3] = -(world_to_camera @ camera_center)

    width = int(image.columns)
    height = int(image.rows)
    multiplier = float(view["spec"].get("ortho_scale_multiplier", 1.0))
    ortho_scale = float(view["camera"]["render_ortho_scale"]) * multiplier
    focal = virtual_distance * max(width, height) / ortho_scale
    intrinsic = np.array(
        [
            [focal, 0.0, width * 0.5],
            [0.0, focal, height * 0.5],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    return intrinsic, extrinsic


def ensure_output_dir(path: Path, overwrite: bool) -> None:
    path.mkdir(parents=True, exist_ok=True)
    if any(path.iterdir()) and not overwrite:
        raise RuntimeError(
            f"Output directory is not empty: {path}. Use --overwrite to replace files."
        )


def normalize_output_material(mesh: o3d.t.geometry.TriangleMesh) -> None:
    """Avoid Open3D's 0.6 base-color and white emissive defaults in GLB."""
    material = mesh.material
    material.vector_properties["base_color"] = np.array(
        [1.0, 1.0, 1.0, 1.0], dtype=np.float32
    )
    material.vector_properties["emissive_color"] = np.array(
        [0.0, 0.0, 0.0, 1.0], dtype=np.float32
    )
    material.scalar_properties["metallic"] = 0.0
    material.scalar_properties["roughness"] = 1.0


def write_calibration(
    output_dir: Path,
    views: list[dict[str, Any]],
    intrinsics: list[np.ndarray],
    extrinsics: list[np.ndarray],
    virtual_distance: float,
) -> None:
    if all(np.allclose(intrinsics[0], matrix) for matrix in intrinsics[1:]):
        calibration_k = intrinsics[0]
    else:
        calibration_k = np.stack(intrinsics)
    np.savez(
        output_dir / "cameras.npz",
        K=calibration_k,
        Rts=np.stack(extrinsics).astype(np.float64),
        images=np.asarray([str(view["image"]) for view in views]),
        labels=np.asarray([view["name"] for view in views]),
        virtual_distance=np.float64(virtual_distance),
        projection_type=np.asarray("virtual_pinhole_from_orthographic"),
    )


def main() -> int:
    args = parse_args()
    dataset = args.dataset.resolve()
    if not dataset.is_dir():
        raise FileNotFoundError(f"Dataset directory not found: {dataset}")

    output_dir = (
        args.output_dir.resolve()
        if args.output_dir is not None
        else dataset / "open3d_out"
    )
    ensure_output_dir(output_dir, args.overwrite)

    mesh_path = find_mesh(dataset, args.mesh)
    metadata = load_metadata(dataset, args.metadata)
    selected_names = None
    if args.views:
        selected_names = [name.strip() for name in args.views.split(",") if name.strip()]
    views = load_views(dataset, metadata, selected_names)

    legacy_mesh, mesh_info = prepare_mesh(
        mesh_path,
        metadata,
        views,
        args.coordinate_frame,
        args.mesh_scale,
        not args.no_weld,
    )
    tensor_mesh = o3d.t.geometry.TriangleMesh.from_legacy(legacy_mesh)
    if "texture_uvs" not in tensor_mesh.triangle:
        raise RuntimeError("Open3D did not convert the OBJ UV atlas to texture_uvs")

    if args.virtual_distance == 0:
        virtual_distance = max(
            1000.0,
            1000.0 * float(mesh_info["mesh_radius"]),
            100.0 * max_original_camera_depth(views),
        )
    else:
        virtual_distance = float(args.virtual_distance)

    images: list[o3d.t.geometry.Image] = []
    intrinsics: list[np.ndarray] = []
    extrinsics: list[np.ndarray] = []
    for view in views:
        image = o3d.t.io.read_image(str(view["image"]))
        if image.columns < 1 or image.rows < 1:
            raise ValueError(f"Could not read image: {view['image']}")
        intrinsic, extrinsic = make_virtual_pinhole(view, image, virtual_distance)
        images.append(image)
        intrinsics.append(intrinsic)
        extrinsics.append(extrinsic)

    write_calibration(output_dir, views, intrinsics, extrinsics, virtual_distance)

    print(f"Mesh: {mesh_path}")
    print(
        "Coordinate conversion: {frame}; scale={scale:.9g}; "
        "vertices={vertices}; triangles={triangles}".format(
            frame=mesh_info["coordinate_frame"],
            scale=mesh_info["mesh_scale"],
            vertices=len(legacy_mesh.vertices),
            triangles=len(legacy_mesh.triangles),
        )
    )
    welding = mesh_info["welding"]
    if welding["enabled"]:
        print(
            "Vertex welding: {removed} duplicate vertices removed; "
            "{output} remain; {degenerate} degenerate faces removed".format(
                removed=welding["removed_duplicate_vertices"],
                output=welding["output_vertices"],
                degenerate=welding["removed_degenerate_triangles"],
            )
        )
    print(f"Views: {len(views)}; virtual pinhole distance: {virtual_distance:.9g}")

    tensor_images = [image for image in images]
    intrinsic_tensors = [Tensor(matrix) for matrix in intrinsics]
    extrinsic_tensors = [Tensor(matrix) for matrix in extrinsics]
    albedo = tensor_mesh.project_images_to_albedo(
        tensor_images,
        intrinsic_tensors,
        extrinsic_tensors,
        args.tex_size,
        True,
    )
    if args.contrast != 1.0:
        albedo = albedo.linear_transform(scale=args.contrast)
    normalize_output_material(tensor_mesh)

    albedo_path = output_dir / "albedo.png"
    if not o3d.t.io.write_image(str(albedo_path), albedo):
        raise RuntimeError(f"Failed to write {albedo_path}")

    mesh_path_out = None
    if not args.no_glb:
        mesh_path_out = output_dir / "mesh_albedo.glb"
        ok = o3d.t.io.write_triangle_mesh(
            str(mesh_path_out),
            tensor_mesh,
            write_vertex_normals=False,
            write_vertex_colors=False,
            write_triangle_uvs=True,
        )
        if not ok:
            raise RuntimeError(f"Failed to write {mesh_path_out}")

    run_meta = {
        "implementation": "open3d_project_to_albedo",
        "open3d_version": o3d.__version__,
        "mesh": str(mesh_path),
        "texture_size": args.tex_size,
        "contrast": args.contrast,
        "coordinate_frame": mesh_info,
        "virtual_distance": virtual_distance,
        "views": [
            {
                "name": view["name"],
                "image": str(view["image"]),
                "render_ortho_scale": float(view["camera"]["render_ortho_scale"]),
                "ortho_scale_multiplier": float(
                    view["spec"].get("ortho_scale_multiplier", 1.0)
                ),
            }
            for view in views
        ],
        "outputs": {
            "albedo": str(albedo_path),
            "mesh": str(mesh_path_out) if mesh_path_out is not None else None,
            "cameras": str(output_dir / "cameras.npz"),
        },
        "note": (
            "Orthographic cameras were converted to virtual pinhole cameras "
            "for Open3D 0.19; increase --virtual-distance for a closer "
            "orthographic approximation."
        ),
    }
    with (output_dir / "run_meta.json").open("w", encoding="utf-8") as stream:
        json.dump(run_meta, stream, indent=2)

    print(f"Albedo: {albedo_path}")
    if mesh_path_out is not None:
        print(f"Textured mesh: {mesh_path_out}")
    print(f"Calibration: {output_dir / 'cameras.npz'}")
    print(f"Run metadata: {output_dir / 'run_meta.json'}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (FileNotFoundError, KeyError, RuntimeError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(2)

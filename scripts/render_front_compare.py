#!/usr/bin/env python3
"""Render front-view base-color comparisons with PyVista.

The renderer is intentionally independent of the C++ texture pipeline.  It loads
the original GLB and the generated OBJ with trimesh, extracts only base-color
textures, and renders both meshes with the same strict orthographic camera.
"""

from __future__ import annotations

import argparse
import gc
import io
import json
import math
import sys
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pyvista as pv
import trimesh
from PIL import Image, ImageDraw, ImageFont


@dataclass(frozen=True)
class ViewCamera:
    right: np.ndarray
    up: np.ndarray
    backward: np.ndarray
    position: np.ndarray
    distance: float
    ortho_scale: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render original GLB and generated OBJ front views side by side."
    )
    parser.add_argument(
        "--dataset",
        required=True,
        help="eval_6cases.zip or an extracted eval_6cases/ dataset root",
    )
    parser.add_argument(
        "--results",
        required=True,
        help="Root directory containing per-case result.obj outputs",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output directory (default: <results>/front_compare)",
    )
    parser.add_argument(
        "--view",
        default="front",
        help="View name to render from render_manifest.json (default: front)",
    )
    parser.add_argument(
        "--size",
        type=int,
        default=1024,
        help="Square render resolution for each side (default: 1024)",
    )
    parser.add_argument(
        "--case",
        action="append",
        default=[],
        help="Case id to render; may be repeated. Defaults to every result case.",
    )
    parser.add_argument(
        "--diff",
        action="store_true",
        help="Additionally write a 4x amplified RGB difference image",
    )
    return parser.parse_args()


def resolve_path(value: str | Path) -> Path:
    return Path(value).expanduser().resolve()


def source_to_render(vertices: np.ndarray) -> np.ndarray:
    """Convert source/GLB coordinates to the render coordinate system."""
    vertices = np.asarray(vertices, dtype=np.float64)
    result = np.empty_like(vertices)
    result[:, 0] = vertices[:, 0]
    result[:, 1] = -vertices[:, 2]
    result[:, 2] = vertices[:, 1]
    return result


class DatasetSource:
    """Read the small subset of eval_6cases needed for rendering."""

    def __init__(self, path: Path, sample_case: str):
        self.path = path
        self.zip_file: zipfile.ZipFile | None = None
        self.prefix = ""

        if path.is_file():
            try:
                self.zip_file = zipfile.ZipFile(path, "r")
            except zipfile.BadZipFile as exc:
                raise ValueError(f"Dataset is not a valid zip file: {path}") from exc

            suffix = f"/{sample_case}/mesh/{sample_case}_textured.glb"
            candidates = [name for name in self.zip_file.namelist() if name.endswith(suffix)]
            if not candidates:
                raise ValueError(f"Could not find original GLB for case {sample_case!r} in {path}")
            candidate = candidates[0]
            self.prefix = candidate[: -len(suffix)]
        elif path.is_dir():
            expected = path / sample_case / "mesh" / f"{sample_case}_textured.glb"
            if not expected.is_file():
                raise ValueError(f"Could not find original GLB: {expected}")
        else:
            raise FileNotFoundError(f"Dataset path does not exist: {path}")

    def close(self) -> None:
        if self.zip_file is not None:
            self.zip_file.close()

    def read_bytes(self, relative_name: str) -> bytes:
        if self.zip_file is not None:
            member = f"{self.prefix}/{relative_name}"
            try:
                with self.zip_file.open(member) as stream:
                    return stream.read()
            except KeyError as exc:
                raise FileNotFoundError(f"Missing dataset member: {member}") from exc

        path = self.path / relative_name
        if not path.is_file():
            raise FileNotFoundError(f"Missing dataset file: {path}")
        return path.read_bytes()

    def load_original_scene(self, case_id: str) -> trimesh.Scene:
        data = self.read_bytes(f"{case_id}/mesh/{case_id}_textured.glb")
        scene = trimesh.load(io.BytesIO(data), file_type="glb", process=False)
        if not isinstance(scene, trimesh.Scene):
            scene = trimesh.Scene(scene)
        return scene

    def load_view_camera(self, case_id: str, view_name: str) -> ViewCamera:
        manifest_data = self.read_bytes(
            f"{case_id}/rank_000/controls/render_manifest.json"
        )
        manifest = json.loads(manifest_data)
        views = manifest.get("views", [])
        view = next((entry for entry in views if entry.get("label") == view_name), None)
        if view is None:
            available = [entry.get("label") for entry in views]
            raise ValueError(
                f"View {view_name!r} not found for case {case_id!r}; available: {available}"
            )

        matrix = np.asarray(view.get("camera_to_world"), dtype=np.float64)
        if matrix.shape != (4, 4):
            raise ValueError(f"Invalid camera_to_world matrix for case {case_id!r}")

        right = matrix[:3, 0]
        up = matrix[:3, 1]
        backward = matrix[:3, 2]
        position = matrix[:3, 3]
        distance = float(manifest.get("camera_fit", {}).get("distance", 2.2))
        ortho_scale = float(view.get("render_ortho_scale", 1.0))

        return ViewCamera(
            right=right,
            up=up,
            backward=backward,
            position=position,
            distance=distance,
            ortho_scale=ortho_scale,
        )


def load_original_meshes(scene: trimesh.Scene) -> list[trimesh.Trimesh]:
    meshes: list[trimesh.Trimesh] = []

    for node_name in scene.graph.nodes_geometry:
        transform, geometry_name = scene.graph[node_name]
        mesh = scene.geometry[geometry_name]
        if not isinstance(mesh, trimesh.Trimesh) or len(mesh.faces) == 0:
            continue

        mesh = mesh.copy()
        mesh.apply_transform(np.asarray(transform, dtype=np.float64))
        mesh.vertices = source_to_render(mesh.vertices)
        meshes.append(mesh)

    if not meshes:
        raise ValueError("Original GLB contains no triangle meshes")

    # The texture pipeline normalizes the source model to height 1.0 in the
    # render coordinate system before fitting the camera.  Apply the same
    # transform here so the original GLB and generated OBJ share a world scale.
    all_vertices = np.concatenate([mesh.vertices for mesh in meshes], axis=0)
    minimum = all_vertices.min(axis=0)
    maximum = all_vertices.max(axis=0)
    center = (minimum + maximum) * 0.5
    height = float(max(maximum[2] - minimum[2], 1e-12))
    scale = 1.0 / height

    for mesh in meshes:
        mesh.vertices = (mesh.vertices - center) * scale

    return meshes


def load_result_meshes(result_obj: Path) -> list[trimesh.Trimesh]:
    scene = trimesh.load(result_obj, process=False)
    if isinstance(scene, trimesh.Scene):
        meshes = [mesh for mesh in scene.geometry.values() if isinstance(mesh, trimesh.Trimesh)]
    else:
        meshes = [scene]

    meshes = [mesh for mesh in meshes if isinstance(mesh, trimesh.Trimesh) and len(mesh.faces) > 0]
    if not meshes:
        raise ValueError(f"Result OBJ contains no triangle meshes: {result_obj}")

    return meshes


def material_color_factor(material: Any) -> np.ndarray:
    raw = None
    for name in ("baseColorFactor", "diffuse"):
        raw = getattr(material, name, None)
        if raw is not None:
            break
    if raw is None:
        return np.ones(4, dtype=np.float64)

    factor = np.asarray(raw, dtype=np.float64)
    if factor.size == 3:
        factor = np.concatenate([factor, [1.0]])
    if factor.size != 4:
        return np.ones(4, dtype=np.float64)
    if float(np.max(factor)) > 1.0 + 1e-6:
        factor = factor / 255.0
    return factor


def material_base_color_image(material: Any) -> Image.Image | None:
    image = None
    for name in ("baseColorTexture", "diffuseTexture", "image"):
        value = getattr(material, name, None)
        if value is not None:
            image = value
            break

    if image is None:
        return None
    if isinstance(image, Image.Image):
        return image.convert("RGB")
    if isinstance(image, (str, Path)):
        return Image.open(image).convert("RGB")
    if isinstance(image, np.ndarray):
        if image.ndim == 3 and image.shape[2] in (3, 4):
            return Image.fromarray(image[:, :, :3]).convert("RGB")
        if image.ndim == 2:
            return Image.fromarray(image).convert("RGB")

    return None


def base_color_texture(mesh: trimesh.Trimesh) -> pv.Texture | None:
    material = getattr(mesh.visual, "material", None)
    image = material_base_color_image(material) if material is not None else None

    if image is None:
        return None

    factor = material_color_factor(material) if material is not None else np.ones(4)
    if not np.allclose(factor[:3], 1.0, atol=1.0 / 255.0):
        pixels = np.asarray(image, dtype=np.float32) * factor[:3].astype(np.float32)
        image = Image.fromarray(
            np.clip(pixels, 0.0, 255.0).astype(np.uint8), mode="RGB"
        )

    texture = pv.Texture(np.asarray(image, dtype=np.uint8))
    texture.use_srgb_color_space = False
    if hasattr(pv.Texture, "WrapType"):
        texture.wrap = pv.Texture.WrapType.CLAMP_TO_EDGE
    else:
        texture.repeat = False
    return texture


def solid_color_texture(mesh: trimesh.Trimesh) -> pv.Texture:
    material = getattr(mesh.visual, "material", None)
    factor = material_color_factor(material) if material is not None else np.ones(4)
    color = np.clip(np.asarray(factor[:3]) * 255.0, 0, 255).astype(np.uint8)
    image = Image.fromarray(np.tile(color, (1, 1, 1)), mode="RGB")
    texture = pv.Texture(np.asarray(image, dtype=np.uint8))
    texture.use_srgb_color_space = False
    return texture


def ensure_uv(mesh: trimesh.Trimesh) -> np.ndarray | None:
    uv = getattr(mesh.visual, "uv", None)
    if uv is None or len(uv) != len(mesh.vertices):
        return None
    return np.asarray(uv, dtype=np.float32)


def mesh_to_pyvista(mesh: trimesh.Trimesh) -> pv.PolyData:
    faces = np.asarray(mesh.faces, dtype=np.int64)
    connectivity = np.hstack(
        [np.full((faces.shape[0], 1), 3, dtype=np.int64), faces]
    ).ravel()
    poly = pv.PolyData(np.asarray(mesh.vertices, dtype=np.float32), connectivity)

    uv = ensure_uv(mesh)
    if uv is not None:
        poly.active_texture_coordinates = uv

    return poly


def set_camera(plotter: pv.Plotter, camera: ViewCamera) -> None:
    focal_point = camera.position - camera.backward * camera.distance
    plotter.camera_position = (
        camera.position.tolist(),
        focal_point.tolist(),
        camera.up.tolist(),
    )
    plotter.camera.parallel_projection = True
    plotter.camera.parallel_scale = camera.ortho_scale / 2.0


def render_asset(
    meshes: Sequence[trimesh.Trimesh], camera: ViewCamera, size: int
) -> Image.Image:
    plotter = pv.Plotter(off_screen=True, window_size=(size, size))
    plotter.set_background("white")

    for mesh in meshes:
        texture = base_color_texture(mesh)
        if texture is None:
            texture = solid_color_texture(mesh)
        poly = mesh_to_pyvista(mesh)
        if poly.active_texture_coordinates is None:
            # Material without UVs: map every vertex to the solid texture pixel.
            poly.active_texture_coordinates = np.zeros(
                (poly.n_points, 2), dtype=np.float32
            )

        plotter.add_mesh(
            poly,
            texture=texture,
            lighting=False,
            pbr=False,
            ambient=1.0,
            diffuse=1.0,
            specular=0.0,
            opacity=1.0,
            smooth_shading=False,
            reset_camera=False,
        )

    try:
        plotter.remove_scalar_bar()
    except Exception:
        pass
    try:
        plotter.remove_bounds_axes()
    except Exception:
        pass

    set_camera(plotter, camera)
    if hasattr(plotter.render_window, "use_srgb_color_space"):
        plotter.render_window.use_srgb_color_space = False

    frame = plotter.screenshot(return_img=True)
    plotter.close()

    pixels = np.asarray(frame)[..., :3]
    return Image.fromarray(np.ascontiguousarray(pixels), mode="RGB")


def image_metrics(original: Image.Image, result: Image.Image) -> dict[str, float]:
    if original.size != result.size or original.mode != result.mode:
        raise ValueError("Rendered images must have the same size and mode")

    a = np.asarray(original, dtype=np.float32)
    b = np.asarray(result, dtype=np.float32)
    delta = a - b

    mae = float(np.mean(np.abs(delta)))
    rmse = float(np.sqrt(np.mean(np.square(delta))))
    psnr = float("inf") if rmse == 0.0 else float(20.0 * math.log10(255.0 / rmse))
    return {"mae": mae, "rmse": rmse, "psnr": psnr}


def compose_side_by_side(
    original: Image.Image, result: Image.Image
) -> Image.Image:
    width = original.width + result.width
    height = max(original.height, result.height)
    combined = Image.new("RGB", (width, height), "white")
    combined.paste(original, (0, 0))
    combined.paste(result, (original.width, 0))
    return combined


def difference_image(original: Image.Image, result: Image.Image) -> Image.Image:
    a = np.asarray(original, dtype=np.float32)
    b = np.asarray(result, dtype=np.float32)
    delta = np.clip(np.abs(a - b) * 4.0, 0.0, 255.0).astype(np.uint8)
    return Image.fromarray(delta, mode="RGB")


def load_default_font(size: int = 18) -> ImageFont.ImageFont:
    try:
        return ImageFont.load_default(size=size)
    except TypeError:
        return ImageFont.load_default()


def make_contact_sheet(
    rows: Sequence[tuple[str, Image.Image, Image.Image]], cell_size: int = 512
) -> Image.Image:
    font = load_default_font(18)
    header_height = 36
    row_height = cell_size + 34
    width = cell_size * 2 + 30
    height = header_height + row_height * max(1, len(rows))

    sheet = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(sheet)

    draw.text((10, 8), "Original GLB", fill="black", font=font)
    draw.text((cell_size + 20, 8), "3DTexture", fill="black", font=font)
    draw.line((0, header_height - 1, width, header_height - 1), fill="black", width=2)

    for index, (case_id, original, result) in enumerate(rows):
        y = header_height + index * row_height
        draw.text((10, y), case_id[:70], fill="black", font=font)
        left = original.resize((cell_size, cell_size), Image.Resampling.BICUBIC)
        right = result.resize((cell_size, cell_size), Image.Resampling.BICUBIC)
        sheet.paste(left, (10, y + 26))
        sheet.paste(right, (cell_size + 20, y + 26))

    return sheet


def discover_result_cases(results_root: Path, requested: Sequence[str]) -> list[str]:
    summary_path = results_root / "summary.json"
    case_ids: list[str] = []

    if summary_path.is_file():
        summary = json.loads(summary_path.read_text())
        case_ids = [str(entry["id"]) for entry in summary.get("cases", []) if "id" in entry]
    else:
        case_ids = sorted(
            path.parent.name for path in results_root.glob("*/result.obj") if path.is_file()
        )

    if requested:
        requested_set = set(requested)
        missing = requested_set.difference(case_ids)
        if missing:
            raise ValueError(f"Requested cases are not present in results: {sorted(missing)}")
        case_ids = [case_id for case_id in case_ids if case_id in requested_set]

    if not case_ids:
        raise ValueError(f"No result cases found in {results_root}")

    return case_ids


def main() -> int:
    args = parse_args()
    if args.size < 1:
        raise ValueError("--size must be a positive integer")

    dataset_path = resolve_path(args.dataset)
    results_root = resolve_path(args.results)
    output_root = (
        resolve_path(args.output)
        if args.output is not None
        else results_root / "front_compare"
    )
    output_root.mkdir(parents=True, exist_ok=True)

    case_ids = discover_result_cases(results_root, args.case)
    dataset = DatasetSource(dataset_path, case_ids[0])

    rendered_rows: list[tuple[str, Image.Image, Image.Image]] = []
    cases_json: list[dict[str, Any]] = []

    try:
        for case_id in case_ids:
            case_output: dict[str, Any] = {
                "id": case_id,
                "success": False,
                "view": args.view,
                "resolution": args.size,
            }
            try:
                result_obj = results_root / case_id / "result.obj"
                if not result_obj.is_file():
                    raise FileNotFoundError(f"Missing result OBJ: {result_obj}")

                camera = dataset.load_view_camera(case_id, args.view)
                original_scene = dataset.load_original_scene(case_id)
                original_meshes = load_original_meshes(original_scene)
                result_meshes = load_result_meshes(result_obj)

                original_image = render_asset(original_meshes, camera, args.size)
                result_image = render_asset(result_meshes, camera, args.size)

                original_path = output_root / f"{case_id}_{args.view}_original.png"
                result_path = output_root / f"{case_id}_{args.view}_result.png"
                combined_path = output_root / f"{case_id}_{args.view}.png"
                original_image.save(original_path)
                result_image.save(result_path)
                compose_side_by_side(original_image, result_image).save(combined_path)

                metrics = image_metrics(original_image, result_image)
                case_output.update(
                    {
                        "success": True,
                        "original_image": str(original_path),
                        "result_image": str(result_path),
                        "comparison_image": str(combined_path),
                        **metrics,
                    }
                )

                if args.diff:
                    diff_path = output_root / f"{case_id}_{args.view}_diff.png"
                    difference_image(original_image, result_image).save(diff_path)
                    case_output["difference_image"] = str(diff_path)

                rendered_rows.append((case_id, original_image, result_image))
                print(
                    f"[{case_id}] MAE={metrics['mae']:.3f} RMSE={metrics['rmse']:.3f} "
                    f"PSNR={metrics['psnr']:.2f} dB"
                )
            except Exception as exc:
                case_output["error"] = str(exc)
                print(f"[{case_id}] ERROR: {exc}", file=sys.stderr)

            cases_json.append(case_output)

            original_scene = None
            original_meshes = []
            result_meshes = []
            gc.collect()
    finally:
        dataset.close()

    if rendered_rows:
        contact_path = output_root / "contact.png"
        make_contact_sheet(rendered_rows).save(contact_path)

    summary = {
        "dataset": str(dataset_path),
        "results": str(results_root),
        "output": str(output_root),
        "view": args.view,
        "resolution": args.size,
        "color_mode": "base_color",
        "lighting": False,
        "normal_maps": False,
        "metallic_roughness": False,
        "case_count": len(cases_json),
        "success_count": sum(1 for case in cases_json if case["success"]),
        "failure_count": sum(1 for case in cases_json if not case["success"]),
        "cases": cases_json,
    }
    (output_root / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")

    if not rendered_rows:
        print("No cases were rendered successfully.", file=sys.stderr)
        return 1

    print(f"Comparison output written to {output_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

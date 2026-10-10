# 3DTexture

This repository contains a C++ implementation of
**"Let There Be Color! Large-Scale Texturing of 3D Reconstructions"** (ECCV 2014).
The maintained Linux + CMake build and executable are pure C++ and do not depend
on OpenCV, Open3D, Python, or uv.  An optional Python/PyVista renderer is
provided only for visual comparison.

## Current status

- Linux CMake build and run scripts are available.
- Perspective `.cam` files from the original project remain supported.
- Strict orthographic projection is supported with an explicit `ORTHO` marker.
- Orthographic visibility uses parallel rays rather than a finite camera center.
- Image decoding/encoding uses bundled stb libraries instead of OpenCV.
- Structured CLI input is supported for TUM, `test04`, and `eval_6cases`.
- Exact XYZ vertex welding is enabled by default for the structured `run` path.
- Texture-page size, color adjustment, MRF weighting, and thread usage can be
  configured with CLI flags or environment variables.
- Unit tests cover orthographic projection, welding, dataset discovery, and GLB loading.
- Input meshes used through the legacy positional CLI are preserved as-is.

## Build

Install the system dependencies:

```bash
./scripts/install_dependencies.sh
```

Configure and build:

```bash
cmake --preset linux-release
cmake --build --preset linux-release
```

Alternative:

```bash
./scripts/build.sh
```

The executable is written to:

```text
build/Release/3DTexture
```

## Run

### Structured CLI

```bash
./build/Release/3DTexture run \
    --dataset <dataset-root-or-case-dir> \
    --output <output-root> \
    [--format auto|tum|test04|eval6cases] \
    [--threads N] \
    [--texture-size N] \
    [--color-lambda F] \
    [--color-average 0|1] \
    [--pairwise-weight F] \
    [--no-weld]
```

The dataset root may be either one case directory or a directory containing
multiple case directories. Each case is written to:

```text
<output-root>/<case-id>/
```

A top-level `summary.json` records per-case success, mesh statistics, patch
count, texture-page count, and runtime.

### TUM example

```bash
./run.sh dataset/rgbd_dataset_freiburg1_room
```

or:

```bash
./build/Release/3DTexture run \
    --dataset dataset/rgbd_dataset_freiburg1_teddy \
    --output /tmp/3dtexture-tum \
    --texture-size 2048
```

### eval_6cases example

```bash
./scripts/run_eval_6cases.sh \
    /path/to/eval_6cases.zip \
    /path/to/output-root \
    --texture-size 2048
```

The script extracts the archive to a temporary directory, discovers all case
directories, runs the pipeline, and writes `summary.json`.

### Optional PyVista front-view comparison

The C++ pipeline remains Python-free.  For visual inspection, a separate
PyVista renderer can compare the original textured GLB with the generated
OBJ/MTL/texture output using the same front-view orthographic camera.

Install the optional renderer environment:

```bash
./scripts/install_pyvista_renderer.sh
```

Run the comparison:

```bash
./scripts/compare_eval6_front.sh /mnt/c/Users/JdSc-local/Downloads/eval_6cases.zip /home/jdsc-local/tmp/3dtexture-eval6-full
```

The wrapper creates:

```text
<results>/front_compare/<case>_front_original.png
<results>/front_compare/<case>_front_result.png
<results>/front_compare/<case>_front.png
<results>/front_compare/contact.png
<results>/front_compare/summary.json
```

Each side is rendered at 1024x1024 with the same strict orthographic camera,
the same world scale, lighting disabled, and only base-color textures.  The
original GLB is normalized to the same height-1 render space used by the
pipeline before comparison.  Normal, metallic, roughness, emissive, and other
PBR maps are intentionally ignored.

Useful options:

```bash
./scripts/compare_eval6_front.sh DATASET RESULTS OUTPUT --case CASE_ID
./scripts/compare_eval6_front.sh DATASET RESULTS OUTPUT --diff
./scripts/compare_eval6_front.sh DATASET RESULTS OUTPUT --size 2048
./scripts/compare_eval6_front.sh DATASET RESULTS OUTPUT --novel-views 8
```

`--novel-views N` generates N evenly distributed viewpoints per case and avoids
the original selected viewpoints.  The minimum separation defaults to 12 degrees
and can be changed with `--novel-min-angle`.

The optional renderer dependencies are installed into `.venv-pyvista` and reuse
the system VTK installation.  They do not affect the C++ build or runtime.

For full usage, output, color, camera, metric, and troubleshooting details, see
[docs/pyvista-front-view-comparison.md](docs/pyvista-front-view-comparison.md).

### Legacy positional CLI

The original positional interface remains available:

```bash
./build/Release/3DTexture <mesh.obj> <rgb_cam_dir> <output-dir>
```

The legacy path intentionally does not apply the new automatic welding step.

## Input formats

### TUM / legacy

A case directory must contain:

```text
mesh.obj
rgb_cam/*.cam
rgb_cam/*.jpg  (or .jpeg / .png)
```

Images are loaded in sorted filename order. Each image must have a `.cam` file
with the same base name.

### test04

A case directory must contain:

```text
eval_metadata.json
mesh.obj
<view-name>.png or <view-name>.jpg
```

The adapter reads `selected_views`, aggregates shard camera manifests, converts
OBJ coordinates to the render coordinate system, infers the normalization scale
from `camera_fit`, and builds strict orthographic views.

### eval_6cases

A case directory must contain:

```text
sample_manifest.json
mesh/<case>_textured.glb
rank_000..003/controls/render_manifest.json
rank_000..003/controls/selected_views.json
composite_<view>.png
```

The adapter merges the 16 selected views across the four rank directories and
uses `<case>_textured.glb` as the geometry source. Existing GLB materials and
textures are ignored.

## Camera file format

All rotations and translations define a world-to-camera transform:

```text
p_cam = R * p_world + t
```

### Perspective

```text
PERSP
tx ty tz
r00 r01 r02
r10 r11 r12
r20 r21 r22
f d0 d1 ppect cx cy
```

The original format stores normalized intrinsics. The program converts `f`,
`cx`, and `cy` to pixel units using the loaded image dimensions.

### Orthographic

```text
ORTHO
tx ty tz
r00 r01 r02
r10 r11 r12
r20 r21 r22
sx sy cx cy
```

Projection is:

```text
u = sx * x_cam + cx
v = sy * y_cam + cy
```

Visibility for orthographic views is tested with rays parallel to the camera
forward direction.

For an orthographic image rendered with an `ortho_scale` in world units:

```text
sx = sy = image_size / ortho_scale
```

The principal point is normally the image center.

## Runtime options

CLI values override environment variables, and environment variables override
defaults.

### Texture-page size

```bash
--texture-size 2048
```

or:

```bash
3DTEXTURE_TEXTURE_SIZE=2048
```

Valid values are 1–16384.

### Seamless color adjustment

```bash
--color-lambda 0.3
--color-average 0
```

or:

```bash
3DTEXTURE_COLOR_LAMBDA=0.3
3DTEXTURE_COLOR_AVERAGE=0
```

### MRF smoothness weight

```bash
--pairwise-weight 5
```

or:

```bash
3DTEXTURE_PAIRWISE_WEIGHT=5
```

### Thread limit

```bash
--threads 1
```

or:

```bash
3DTEXTURE_THREADS=1
```

## Vertex welding

The structured `run` path performs exact XYZ welding by default:

- vertices with identical coordinates are merged;
- face indices are remapped;
- degenerate faces are removed;
- unreferenced vertices are removed;
- per-corner normals are retained or generated when missing.

Use `--no-weld` to disable it.

The legacy positional CLI does not weld input meshes.

## Tests

```bash
ctest --test-dir build/Release --output-on-failure
```

The suite covers:

- orthographic projection parsing and math;
- parallel-ray occlusion;
- exact vertex welding;
- dataset discovery;
- minimal GLB geometry loading.

## Implementation notes

The maintained pipeline is:

```text
discover dataset cases
load mesh and views
weld duplicate vertices
build face adjacency
choose per-face source views with an MRF solver
solve seamless color adjustment
pack texture patches
rasterize projected colors into texture pages
write OBJ/MTL/texture outputs
write run_manifest.json and summary.json
```

Patch-packing failures are reported and terminate the affected case.

## Dependencies

System dependencies:

- C++17 compiler
- CMake 3.21 or newer for the supplied presets
- pkg-config
- oneTBB
- unzip for `scripts/run_eval_6cases.sh`

Bundled dependencies:

- Eigen
- stb
- libacc
- mapMAP
- nlohmann/json
- TinyGLTF

There is no OpenCV or Open3D dependency.

Optional visual-comparison dependencies:

- Python 3.10 or newer
- System VTK (`python3-vtk` on Ubuntu)
- PyVista, trimesh, pygltflib, NumPy, Pillow
- `xvfb-run` when rendering on a headless Linux system

The optional Python environment is created by
`scripts/install_pyvista_renderer.sh` and is not required by the C++ pipeline.

## Directory layout

```text
CMakeLists.txt              CMake build description
CMakePresets.json           Linux debug/release presets
docs/                       PyVista comparison documentation
scripts/                    Install, build, run, eval_6cases, and PyVista compare scripts
src/                        C++ implementation
tests/                      Unit and integration tests
thirdparty/                 Bundled C++ dependencies
dataset/                    Bundled TUM examples
3DTexture/                  Legacy Visual Studio project
x64/                         Legacy Windows example binaries
```

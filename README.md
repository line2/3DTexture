# 3DTexture

This repository contains a C++ implementation of
**"Let There Be Color! Large-Scale Texturing of 3D Reconstructions"** (ECCV 2014).
The current maintained build is Linux + CMake, and the executable is pure C++.
It does not depend on OpenCV, Open3D, Python, or uv.

## Current status

- Linux CMake build and run scripts are available.
- Perspective `.cam` files from the original project remain supported.
- Strict orthographic projection is supported with an explicit `ORTHO` marker.
- Orthographic visibility uses parallel rays rather than a finite camera center.
- Image decoding/encoding uses bundled stb libraries instead of OpenCV.
- Texture-page size, color adjustment, MRF weighting, and thread usage can be
  configured with environment variables.
- A unit test covers orthographic projection parsing, projection math, and
  parallel-ray visibility.
- Input meshes are used as provided. This version does not automatically weld
  duplicate vertices or convert an external metadata format such as
  `eval_metadata.json` into `.cam` files.

## Build

Install the system dependencies:

```bash
./scripts/install_dependencies.sh
```

Configure and build with the Linux release preset:

```bash
cmake --preset linux-release
cmake --build --preset linux-release
```

Alternatively, use the build script:

```bash
./scripts/build.sh
```

The executable is written to:

```text
build/Release/3DTexture
```

Debug builds are also available:

```bash
cmake --preset linux-debug
cmake --build --preset linux-debug
```

## Run

Run the default bundled TUM dataset:

```bash
./run.sh
```

Run another bundled dataset:

```bash
./run.sh dataset/rgbd_dataset_freiburg1_room
```

Call the executable directly:

```bash
./build/Release/3DTexture <mesh.obj> <rgb_cam_dir> <output_dir>
```

For example:

```bash
./build/Release/3DTexture \
    dataset/rgbd_dataset_freiburg1_teddy/mesh.obj \
    dataset/rgbd_dataset_freiburg1_teddy/rgb_cam \
    dataset/rgbd_dataset_freiburg1_teddy/out
```

The output directory is created if necessary. Typical outputs are:

```text
result.obj
result.mtl
label.obj
label.txt
0000.png
0001.png
...
```

## Input format

A dataset directory must contain:

```text
mesh.obj
rgb_cam/*.cam
rgb_cam/*.jpg  (or .jpeg / .png)
```

Each image must have a `.cam` file with the same base name. Images are loaded
in sorted filename order. For stable results, use numeric prefixes such as:

```text
00_front.jpg  00_front.cam
01_right.jpg  01_right.cam
```

The executable expects the mesh and camera poses to already be in the same
coordinate system. It does not read `eval_metadata.json`, perform an OBJ
coordinate conversion, or choose views automatically.

## Camera file format

All rotations and translations define a world-to-camera transform:

```text
p_cam = R * p_world + t
```

### Perspective cameras

A legacy `.cam` file without a projection marker is interpreted as a
perspective camera. An explicit marker is also accepted:

```text
PERSP
tx ty tz
r00 r01 r02
r10 r11 r12
r20 r21 r22
f d0 d1 ppect cx cy
```

The original format stores normalized intrinsic values. The program converts
`f`, `cx`, and `cy` to pixel units using the loaded image dimensions.

### Orthographic cameras

Orthographic views must start with an explicit marker:

```text
ORTHO
tx ty tz
r00 r01 r02
r10 r11 r12
r20 r21 r22
sx sy cx cy
```

The intrinsics are already in pixel units. Projection is:

```text
u = sx * x_cam + cx
v = sy * y_cam + cy
```

Visibility for orthographic views is tested with rays parallel to the camera
forward direction. This avoids incorrectly using a finite perspective camera
center for an orthographic render.

For an orthographic image rendered with an `ortho_scale` in world units, the
usual scale conversion is:

```text
sx = sy = image_size / ortho_scale
```

The principal point is normally the image center.

## Runtime options

### Texture-page size

The default square texture-page size is 1000 pixels. Meshes that have been
welded can contain large connected patches and may require a larger page:

```bash
3DTEXTURE_TEXTURE_SIZE=2048 ./build/Release/3DTexture mesh.obj views out
```

Valid values are 1–16384.

### Seamless color adjustment

The seamless color solver exposes two options:

```bash
3DTEXTURE_COLOR_LAMBDA=0.3
3DTEXTURE_COLOR_AVERAGE=0
```

- `3DTEXTURE_COLOR_LAMBDA` controls the smoothness weight. Larger values
  favor smoother adjustments; smaller values enforce stronger seam agreement.
- `3DTEXTURE_COLOR_AVERAGE=0` uses direct projected vertex colors instead of
  the sampled edge average.

The defaults are:

```text
3DTEXTURE_COLOR_LAMBDA=0.1
3DTEXTURE_COLOR_AVERAGE=1
```

### MRF smoothness weight

The Potts smoothness weight can be overridden:

```bash
3DTEXTURE_PAIRWISE_WEIGHT=5
```

The default is `1`.

### Thread limit

MapMAP, oneTBB, and BVH construction can be limited to a fixed number of
threads:

```bash
3DTEXTURE_THREADS=1
```

Omitting the variable uses the hardware thread count. A value of `1` is useful
for deterministic debugging and for comparing runs.

## Tests

After building, run the CTest suite:

```bash
ctest --test-dir build/Release --output-on-failure
```

The current test covers:

- explicit orthographic `.cam` parsing,
- orthographic projection values,
- parallel-ray visibility,
- backward compatibility with unmarked perspective `.cam` files.

## Implementation notes

The maintained pipeline is:

```text
load OBJ
load images and .cam poses
build face adjacency
choose per-face source views with an MRF solver
solve seamless color adjustment
pack texture patches
rasterize projected colors into texture pages
write OBJ/MTL/texture outputs
```

Patch packing failures are reported and terminate the run. If a welded input
mesh contains a patch larger than the configured page, increase
`3DTEXTURE_TEXTURE_SIZE`.

## Dependencies

Runtime/build dependencies:

- C++17 compiler
- CMake (3.21 or newer is recommended for the supplied presets)
- pkg-config
- oneTBB from the system package manager

Bundled source dependencies:

- Eigen for matrix calculations and linear solves
- stb for JPEG/PNG decoding and PNG writing
- libacc for BVH-accelerated occlusion tests
- mapMAP for MRF optimization

There is no OpenCV or Open3D dependency.

## Directory layout

```text
CMakeLists.txt              CMake build description
CMakePresets.json           Linux debug/release presets
scripts/install_dependencies.sh
scripts/build.sh
run.sh
src/                         C++ implementation
tests/                       Orthographic projection unit test
thirdparty/                  Eigen, stb, libacc, and mapMAP sources
dataset/                     Bundled TUM examples
3DTexture/                   Legacy Visual Studio project
x64/                         Legacy Windows example binaries
```

The legacy Windows project files are retained for reference. CMake is the
maintained build orchestrator.

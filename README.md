# 3DTexture

This code is an implementation of "Let There Be Color! Large-Scale Texturing of 3D Reconstructions"(ECCV2014).

The project now supports Linux through CMake. On a Linux host, install the required packages and build with:

```bash
./scripts/install_dependencies.sh
cmake --preset linux-release
cmake --build --preset linux-release
./run.sh
```

`run.sh` builds the Release executable if necessary and processes the default TUM example dataset. To process another bundled dataset:

```bash
./run.sh dataset/rgbd_dataset_freiburg1_room
```

The executable can also be called directly:

```bash
./build/Release/3DTexture <mesh.obj> <rgb_cam_dir> <output_dir>
```

### Camera projection models

Legacy `.cam` files without a leading marker are parsed as perspective cameras. An explicit perspective marker is also supported:

```text
PERSP
tx ty tz
r00 r01 r02
r10 r11 r12
r20 r21 r22
f d0 d1 ppect cx cy
```

Orthographic cameras use a leading marker and pixel-unit intrinsics:

```text
ORTHO
tx ty tz
r00 r01 r02
r10 r11 r12
r20 r21 r22
sx sy cx cy
```

For an orthographic view, projection is:

```text
u = sx * x_cam + cx
v = sy * y_cam + cy
```

Orthographic visibility is evaluated with parallel rays instead of a finite perspective camera center.

### Color adjustment tuning

The seamless color solver exposes two runtime options:

```bash
3DTEXTURE_COLOR_LAMBDA=0.3
3DTEXTURE_COLOR_AVERAGE=0
```

`3DTEXTURE_COLOR_LAMBDA` controls the smoothness weight. Larger values favor smoother adjustments; smaller values enforce stronger seam agreement. `3DTEXTURE_COLOR_AVERAGE=0` uses the direct projected vertex color instead of the sampled edge average. On the bundled orthographic test, `0.3` and direct vertex colors produced the most consistent seams.

Set a larger square texture page when welded meshes produce large connected patches:

```bash
3DTEXTURE_TEXTURE_SIZE=2048
```

The default remains 1000. Valid values are 1–16384.

Both variables are optional; the defaults remain `0.1` and edge averaging enabled.

The MRF Potts smoothness weight can also be overridden:

```bash
3DTEXTURE_PAIRWISE_WEIGHT=5
```

The default is `1`.

Limit both mapMAP/oneTBB and BVH construction to a fixed number of threads:

```bash
3DTEXTURE_THREADS=1
```

Omitting the variable uses the hardware thread count.

Equivalent build/run scripts are kept for Windows, while CMake is now the build orchestrator for Linux.

Directory structure:

./CMakeLists.txt and ./CMakePresets.json: Linux CMake build.

./scripts/install_dependencies.sh: installs CMake, C++ compiler, pkg-config, and oneTBB for supported Linux package managers.

./scripts/build.sh: configures and builds with CMake.

./run.sh: builds and runs the default dataset.

./3Dtexture: legacy Visual Studio project.

./dataset: bundled TUM test datasets containing meshes, RGB images, and camera pose files.

./src: main source code of this project.

./thirdparty: open source libraries on which this project depends.
		- Eigen: for basic matrix calculation and solving linear equations.
		- stb: JPEG/PNG decoding and PNG writing.
		- libacc: for 3D occlusion culling.
		- mapMAP: as an MRF solver (graph cut).
		- tbb/oneTBB: parallel computing, used by mapMAP.

The Linux build uses the bundled Eigen, libacc, mapMAP, and stb sources from `thirdparty`; oneTBB is expected from the system package manager. Image loading and PNG writing use stb instead of OpenCV.

./x64: example EXEs and DLLs.

./run.bat: one-key to run executable file.



## Open3D albedo implementation (uv environment)

A second, Python implementation is available under `tools/open3d_project_to_albedo.py`.
It follows Open3D's `TriangleMesh::project_images_to_albedo` workflow from the
upstream `triangle_mesh_project_to_albedo.py` example, while adding an adapter
for the `test_04` orthographic-camera metadata.

Install the locked Python environment and run it with:

```bash
./scripts/install_python_dependencies.sh
./scripts/run_open3d_albedo.sh /path/to/test_04
```

Or directly:

```bash
uv run --locked tools/open3d_project_to_albedo.py /path/to/test_04
```

The tool writes:

```text
test_04/open3d_out/albedo.png
test_04/open3d_out/mesh_albedo.glb
test_04/open3d_out/cameras.npz
test_04/open3d_out/run_meta.json
```

Important implementation details:

- The input OBJ must contain an existing UV atlas; Open3D requires
  `texture_uvs`.
- `eval_metadata.json` stores camera-to-world matrices whose columns are
  image-right, image-up, and camera-backward. The tool converts these to
  Open3D's world-to-camera convention using image-down as row 1 and
camera-forward as row 2, while preserving the original image-plane origin.
- Open3D 0.19 has no orthographic overload for
  `project_images_to_albedo`. Each ORTHO camera is therefore converted to an
  equivalent pinhole camera at a large virtual distance:
  `f = virtual_distance * image_size / render_ortho_scale`.
  The default is a conservative large distance based on mesh radius and original camera depth (at least `1000 * mesh_radius`). Increase
  `--virtual-distance` if you need a closer orthographic approximation.
- The uniform render mesh scale is inferred from `camera_fit.per_view` when it
  is available. For the raw `test_04` OBJ this recovers the `1 / 1.8`
  normalization automatically. `--mesh-scale` and `--coordinate-frame` can
  override this behavior.
- Output texture size is controlled by `--tex-size` (default 2048).
- The GLB material is normalized to a white base color, zero emissive color,
  zero metallic, and roughness 1. This prevents Open3D's default 0.6 base color
  from darkening the exported albedo.


Run the Python camera-conversion test with:

```bash
./scripts/test_python.sh
```


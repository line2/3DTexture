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

Equivalent build/run scripts are kept for Windows, while CMake is now the build orchestrator for Linux.

Directory structure:

./CMakeLists.txt and ./CMakePresets.json: Linux CMake build.

./scripts/install_dependencies.sh: installs CMake, C++ compiler, pkg-config, OpenCV, and oneTBB for supported Linux package managers.

./scripts/build.sh: configures and builds with CMake.

./run.sh: builds and runs the default dataset.

./3Dtexture: legacy Visual Studio project.

./dataset: bundled TUM test datasets containing meshes, RGB images, and camera pose files.

./src: main source code of this project.

./thirdparty: open source libraries on which this project depends.
		- Eigen: for basic matrix calculation and solving linear equations.
		- Opencv: linked from the system installation.
		- libacc: for 3D occlusion culling.
		- mapMAP: as an MRF solver (graph cut).
		- tbb/oneTBB: parallel computing, used by mapMAP.

The Linux build uses the bundled Eigen, libacc, and mapMAP sources from `thirdparty`, while OpenCV and oneTBB are expected from the system package manager.

./x64: example EXEs and DLLs.

./run.bat: one-key to run executable file.



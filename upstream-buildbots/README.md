# Upstream Buildbot Docker Recipes

This folder contains the different Dockerfiles that serve as the basis for the LLVM upstream-buildbots that we maintain.
These files can be used to recreate the docker container images and allow a developer to reproduce build and potentially test issues locally.
During container build time certain ROCm components are pulled-in.
Depending on the respective container / OS, this may result in a large container image.

## ROCm installation (manylinux images)

The `manylinux-build-only` and `manylinux-hip-tpl` images install ROCm from a
[TheRock](https://github.com/ROCm/TheRock) nightly "dist" tarball rather than
from packages.
The full dist tarball contains the complete ROCm SDK (HIP runtime, device libs,
`rocminfo`, `rocblas`, `rocthrust`, ...) and is extracted into `/opt/rocm` by the
`install-rocm-nightly.sh` helper script in each image directory.

By default the most recent nightly matching the configured base version and gfx
target is installed. The behavior is controlled via build args:

| Build arg           | Default  | Meaning                                                                                         |
| ------------------- | -------- | ----------------------------------------------------------------------------------------------- |
| `ROCM_BASE_VERSION` | `7.14`   | Base ROCm version to track.                                                                     |
| `ROCM_GFX`          | `gfx90a` | gfx target family.                                                                              |
| `ROCM_NIGHTLY_DATE` | (empty)  | Pin a specific build date `YYYYMMDD`; empty auto-detects the latest. Must be 8 digits when set. |

To pin a reproducible build, pass the date at build time:

```
sudo docker build --build-arg ROCM_NIGHTLY_DATE=20260610 -t <image> -f Dockerfile .
```

The helper records what was actually installed in `/opt/rocm/.info/nightly`
(tarball name, URL, base version, gfx target, and date), complementing TheRock's
own `/opt/rocm/.info/version`, which only carries the base version and cannot
distinguish nightlies.

We build the containers with a docker invocation adjacent to

```
cd Ubu22
sudo docker build -t upstreamimages/<OS>/<version>:date -f Dockerfile .
```

For starting the container we use different CPU sets to place multiple containers on a single physical node.
A manual start of the container should look similar to

```
sudo docker run --rm -it --network=host --device=/dev/kfd --device=/dev/dri --group-add video --cpuset-cpus 0-31 --user botworker <container-image> bash
```

## Quick setup with `run.py`

`run.py` is a helper that lets a developer quickly set up an environment from one
of the manylinux image directories (e.g. `manylinux-build-only`, `manylinux-hip-tpl`)
for local debugging and reproduction. It downloads the base build context from
[TheRock](https://github.com/ROCm/TheRock/tree/main/dockerfiles), builds the
images, and creates a container ready to `docker exec` into.

### Usage

```
python run.py <target> [--pull] [--build] [--clean] [options]
```

`<target>` is one of `manylinux-build-only` or `manylinux-hip-tpl`.

When no operation flag is given, the default flow is **pull then build** (download
the base files, build the images, and start the container). The operations run in
a fixed order (`clean` -> `pull` -> `build`) and abort if any step fails.

| Operation | Meaning                                                                                                                                       |
| --------- | --------------------------------------------------------------------------------------------------------------------------------------------- |
| `--pull`  | Download `build_manylinux_x86_64.Dockerfile` and the helper scripts it needs into `<dest>/manylinux-base`.                                     |
| `--build` | Build the base image `localhost/manylinux:base` (only if missing), build the target image (tagged `<target>`), then create and run a container. |
| `--clean` | Remove the container, the target image, and the base image.                                                                                   |

| Option     | Default        | Meaning                                                              |
| ---------- | -------------- | -------------------------------------------------------------------- |
| `--dest`   | current dir    | Where the base build context (`manylinux-base/`) is downloaded/read. |
| `--name`   | `test-<target>` | Container name.                                                      |
| `--no-gpu` | (off)          | Skip the GPU device/group run flags when starting the container.     |
| `--llvm-src` | (none)       | Bind-mount a local LLVM source tree at `/home/botworker/bbot/llvm-project`. |

`--dest` is not remembered between runs. If you split `--pull` and `--build` into
two separate invocations and pulled to a non-default location, you must pass the
same `--dest` to `--build` so it can find the downloaded base files. For example,
after `python run.py manylinux-build-only --pull --dest ~/test`, build with:

```
python run.py manylinux-build-only --build --dest ~/test
```

Omitting `--dest` on the build step would look in `./manylinux-base` instead and
fail with a "Run a pull first" error (the base image is only built by `--build`,
not by `--pull`).

Use `--llvm-src` to mount an existing local LLVM checkout instead of cloning
inside the container; it appears at `/home/botworker/bbot/llvm-project`.

The container is started detached and kept alive, so you can open a shell with:

```
docker exec -it test-<target> bash
```

### Examples

```
# Default: pull base files, build the images, and start the container
python run.py manylinux-build-only

# Only download the base Dockerfile + helper scripts
python run.py manylinux-build-only --pull

# Split pull and build into two steps with a custom location
# (pass the same --dest to both so build can find the pulled files)
python run.py manylinux-build-only --pull --dest ~/test
python run.py manylinux-build-only --build --dest ~/test

# Build and run without GPU device flags
python run.py manylinux-build-only --build --no-gpu

# Mount a local LLVM source tree instead of cloning inside the container
python run.py manylinux-hip-tpl --build --llvm-src ~/git/llvm-project

# Remove the container and images
python run.py manylinux-build-only --clean
```

## Assumptions / Requirements

- The images require a working AMDGPU dkms / KFD to be installed in order to test work on the GPU.
- The images assume a group id for the `render` group of `109`.
- This is currently hardcoded in the Dockerfile.
  If this does apply on your system, please go ahead and change that accordingly.
  Check the group id on your machine via `cat /etc/group | grep render`.
- The CMake cache file sets individual timeouts per test case.
  This requires the `psutil` Python module to be installed, which is done through ansible on the buildbots.
  When running the container manually, you can install it via `python3 -m pip install psutil`.

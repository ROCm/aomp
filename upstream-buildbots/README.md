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

## Assumptions / Requirements

- The images require a working AMDGPU dkms / KFD to be installed in order to test work on the GPU.
- The images assume a group id for the `render` group of `109`.
- This is currently hardcoded in the Dockerfile.
  If this does apply on your system, please go ahead and change that accordingly.
  Check the group id on your machine via `cat /etc/group | grep render`.
- The CMake cache file sets individual timeouts per test case.
  This requires the `psutil` Python module to be installed, which is done through ansible on the buildbots.
  When running the container manually, you can install it via `python3 -m pip install psutil`.

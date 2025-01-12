# Upstream Buildbot Docker Recipes

This folder contains the different Dockerfiles that serve as the basis for the LLVM upstream-buildbots that we maintain.
These files can be used to recreate the docker container images and allow a developer to reproduce build and potentially test issues locally.
During container build time certain ROCm components are pulled-in.
Depending on the respective container / OS, this may result in a large container image.

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

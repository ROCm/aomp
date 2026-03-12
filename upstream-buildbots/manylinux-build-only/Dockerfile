# Based on TheRock manylinux.
# To use this dockerfile, please first build the base image as localhost/manyliunx:base 
# from the following dockerfile:
# https://github.com/ROCm/TheRock/blob/main/dockerfiles/build_manylinux_x86_64.Dockerfile
FROM localhost/manylinux:base

# Add ROCm repository.
COPY rocm.repo /etc/yum.repos.d/rocm.repo
RUN yum clean packages && yum clean all

# Install minimal ROCm components for buildbot.
RUN yum install -y \
    rocm-device-libs \
    rocm-core \
    rocminfo \
    hsa-rocr-devel7.1.1 \
    && yum clean all && \
    rm -rf /var/cache/yum

# Update render group GID to match host for GPU access.
RUN groupmod -g 109 render

# ROCm library path for offload execution.
ENV LD_LIBRARY_PATH=/opt/rocm/lib:$LD_LIBRARY_PATH

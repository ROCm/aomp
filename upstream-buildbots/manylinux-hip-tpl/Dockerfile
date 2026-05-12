# Based on theRock manylinux.
# To use this dockerfile, please first build the base image as localhost/manyliunx:base
# from the following dockerfile:
# https://github.com/ROCm/TheRock/blob/main/dockerfiles/build_manylinux_x86_64.Dockerfile
FROM localhost/manylinux:base

ARG WORKBASE="/opt"

# Add ROCm repository.
COPY rocm.repo /etc/yum.repos.d/rocm.repo
RUN yum clean packages && yum clean all

# Install minimal ROCm components for buildbot.
RUN yum install -y \
    rocm-device-libs \
    rocm-core \
    rocminfo \
    hsa-rocr-devel7.1.1 \
    rocm-hip-runtime7.1.1.x86_64 \
    rocm-hip-runtime-devel7.1.1.x86_64 \
    rocthrust-devel \
    rocblas \
    && yum clean all && \
    rm -rf /var/cache/yum

# Update render group GID to match host for GPU access.
RUN groupmod -g 109 render

# Create botworker user.
RUN useradd --create-home -G video,render --shell /bin/bash botworker

RUN mkdir -p ${WORKBASE}/botworker/llvm/External/hip

WORKDIR ${WORKBASE}/botworker/llvm/External
RUN cd hip && ln -s /opt/rocm* .

WORKDIR ${WORKBASE}/botworker/llvm
# Clone LLVM test suite
RUN git clone https://github.com/llvm/llvm-test-suite.git
RUN cd .. && chown -R botworker:botworker llvm

# ROCm library path for offload execution.
ENV LD_LIBRARY_PATH=/opt/rocm/lib:$LD_LIBRARY_PATH

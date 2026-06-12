# Based on theRock manylinux.
# To use this dockerfile, please first build the base image as localhost/manyliunx:base
# from the following dockerfile:
# https://github.com/ROCm/TheRock/blob/main/dockerfiles/build_manylinux_x86_64.Dockerfile
FROM localhost/manylinux:base

ARG WORKBASE="/opt"

# Install ROCm from a TheRock nightly dist tarball.
# The full dist tarball already contains the HIP runtime, rocblas, rocthrust,
# etc., so no extra ROCm packages are needed here.
# By default the most recent nightly for the given base version + gfx is used;
# pin a specific build with --build-arg ROCM_NIGHTLY_DATE=YYYYMMDD.
ARG ROCM_BASE_VERSION=7.14
ARG ROCM_GFX=gfx90a
ARG ROCM_NIGHTLY_DATE=

COPY install-rocm-nightly.sh /tmp/install-rocm-nightly.sh
RUN ROCM_BASE_VERSION="${ROCM_BASE_VERSION}" \
    ROCM_GFX="${ROCM_GFX}" \
    ROCM_NIGHTLY_DATE="${ROCM_NIGHTLY_DATE}" \
    bash /tmp/install-rocm-nightly.sh && \
    rm -f /tmp/install-rocm-nightly.sh

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

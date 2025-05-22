# Dockerfile for HIP buildbot container setup

FROM ubuntu:jammy

ARG WORKBASE="/home"

# Install prereqs
COPY prerequisites /tmp/prerequisites

ARG DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y unzip $(cat /tmp/prerequisites)

# Prepare ROCm installation
RUN mkdir --parents --mode=0755 /etc/apt/keyrings
RUN wget https://repo.radeon.com/rocm/rocm.gpg.key -O - | \
    gpg --dearmor | sudo tee /etc/apt/keyrings/rocm.gpg > /dev/null
RUN apt-get update

COPY rocm.list /etc/apt/sources.list.d/rocm.list
COPY rocm-pin-600 /etc/apt/preferences.d/rocm-pin-600
RUN apt-get update

# hsa-rocr-dev brings in the CMake files for libhsa (libxrender required for HIP blender testing)
RUN apt-get install -y rocm-device-libs rocm-core rocminfo hsa-rocr-dev rocm-hip-runtime \
    rocm-hip-runtime-dev libxrender1 libxxf86vm1 libxfixes3 libxi6 libsm6 libice6 libgl1 \
    libxkbcommon0 rocthrust-dev rocblas

# Update the group id inside the container to match host (and get access to GPU)
RUN groupadd --gid 109 render
RUN groupmod -g 109 render

# Create buildbot worker user inside the container
RUN useradd --create-home -G video,render --shell /bin/bash botworker

# Prepare container for HIP buildbot (Kokkos / Ginkgo)
RUN mkdir -p ${WORKBASE}/botworker/llvm/External/hip

WORKDIR ${WORKBASE}/botworker/llvm/External
RUN cd hip && ln -s /opt/rocm* .

WORKDIR ${WORKBASE}/botworker/llvm
# Clone LLVM test suite
RUN git clone https://github.com/llvm/llvm-test-suite.git
RUN cd .. && chown -R botworker:botworker llvm

# XXX: When mounting a volume as home, this does not work and needs to be re-done.
ENV PATH=/home/botworker/.local/bin:$PATH

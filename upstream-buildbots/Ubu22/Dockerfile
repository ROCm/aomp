FROM ubuntu:jammy

# Install prereqs
COPY prerequisites /tmp/prerequisites

ARG DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y $(cat /tmp/prerequisites)

# Prepare ROCm installation
RUN mkdir --parents --mode=0755 /etc/apt/keyrings
RUN wget https://repo.radeon.com/rocm/rocm.gpg.key -O - | \
    gpg --dearmor | sudo tee /etc/apt/keyrings/rocm.gpg > /dev/null
RUN apt-get update
COPY rocm.list /etc/apt/sources.list.d/rocm.list
COPY rocm-pin-600 /etc/apt/preferences.d/rocm-pin-600
RUN apt-get update
# rocminfo brings in libhsa. hsa-rocr-dev brings in the libhsa cmake targets
RUN apt-get install -y rocm-device-libs rocm-core rocminfo hsa-rocr-dev

# Update the group id inside the container to match host (and get access to GPU)
RUN groupadd --gid 109 render
RUN groupmod -g 109 render

# Create buildbot worker user inside the container
RUN useradd --create-home -G video,render --shell /bin/bash botworker

# This is required for offload tests to find libhsa and function correctly
ENV LD_LIBRARY_PATH=/opt/rocm/lib:$LD_LIBRARY_PATH

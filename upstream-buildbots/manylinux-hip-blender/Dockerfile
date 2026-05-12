# Based on TheRock manylinux.
# To use this dockerfile, please first build the base image as localhost/manyliunx:base
# from the following dockerfile:
# https://github.com/ROCm/TheRock/blob/main/dockerfiles/build_manylinux_x86_64.Dockerfile
# Make sure ref.tgz and scene.tgz exist in ./manylinux-hip-blender/
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
    hipify-clang7.1.1.x86_64 \
    && yum clean all && \
    rm -rf /var/cache/yum

# Update render group GID to match host for GPU access.
RUN groupmod -g 109 render

# Create botworker user.
RUN useradd --create-home -G video,render --shell /bin/bash botworker

# The bot expects blender 4.1.1 to be avail as blender in $SOME_BASE/External/hip
# and a folder with scene files in $SOME_BASE/External/hip/Blender_Scenes
RUN mkdir -p ${WORKBASE}/botworker/llvm/External/hip/Blender_Scenes

WORKDIR ${WORKBASE}/botworker/llvm/External
RUN mkdir downloads && cd downloads && curl -O https://download.blender.org/release/Blender4.1/blender-4.1.1-linux-x64.tar.xz
RUN cd hip && tar xf ../downloads/blender-4.1.1-linux-x64.tar.xz && ln -s blender-4.1.1-linux-x64 blender
RUN cd hip && ln -s /opt/rocm* .
RUN touch hip/Blender_Scenes/scenes.txt

# Move the blender test files into the container
WORKDIR ${WORKBASE}/botworker/llvm
COPY ref.tgz External/hip/Blender_Scenes
RUN cd External/hip/Blender_Scenes && tar xzf ref.tgz

COPY scenes.tgz External/hip/Blender_Scenes
RUN cd External/hip/Blender_Scenes && tar xzf scenes.tgz
RUN cd External/hip/Blender_Scenes && mv barbershop_interior_gpu.blend barbershop.blend

WORKDIR ${WORKBASE}/botworker/llvm/External/hip/Blender_Scenes
RUN echo "290skydemo_release" >> scenes.txt
# RUN echo "barbershop_interior_gpu" >> scenes.txt
RUN echo "flat-archiviz" >> scenes.txt
RUN echo "laundromat" >> scenes.txt
RUN echo "monster_under_the_bed_sss_demo_by_metin_seven" >> scenes.txt
RUN echo "pabellon_barcelona/pavillon_barcelone_gpu" >> scenes.txt

WORKDIR ${WORKBASE}/botworker/llvm
# Clone LLVM test suite
RUN git clone https://github.com/llvm/llvm-test-suite.git
RUN cd .. && chown -R botworker:botworker llvm

# When mounting a volume as home, this does not work and needs to be re-done.
ENV PATH=/home/botworker/.local/bin:$PATH
ENV BUILDBOT_ROOT=/home/botworker/bbot/hip-blender
ENV EXTERNAL_DIR=/opt/botworker/llvm/External

#! /usr/bin/env bash

# run_miniQMC.sh - runs several MiniQMC binaries
# The user can set different variables to control execution.
# MQMC_BUILD_PREFIX       miniqmc build prefix (default: ~/miniqmc_build)
# MQMC_SOURCE_DIR         path to miniqmc sources (default: .)
# MQMC_OMP_NUM_THREADS    how many OpenMP threads should be used (default 64)
# MQMC_NUM_BUILD_PROCS    how many processes to build miniqmc (default 32)
# ROCM_INSTALL_PATH       top-level ROCm install directory (default: /opt/rocm-5.3.0)

# --- Start standard header to set AOMP environment variables ----
realpath=`realpath $0`
thisdir=`dirname $realpath`
export AOMP_USE_CCACHE=0

. $thisdir/aomp_common_vars
# --- end standard header ----

# Default ROCm installation
: ${ROCM:=/opt/rocm-5.3.0}

# Control how many OpenMP threads are used by MiniQMCPack
: ${MQMC_OMP_NUM_THREADS:=32}

export PATH=$AOMP/bin:$PATH
#export PATH=/home/janplehr/rocm/trunk/bin:$PATH

# We export all these paths, so they will be picked-up in the CMake command.
export hsaruntime64_DIR=${ROCM}/lib/cmake/hsa-runtime64/
export hipblas_DIR=${ROCM}/lib/cmake/hipblas/
export hip_DIR=${ROCM}/lib/cmake/hip
export AMDDeviceLibs_DIR=${ROCM}/lib/cmake/AMDDeviceLibs/
export amd_comgr_DIR=${ROCM}/lib/cmake/amd_comgr/
export rocblas_DIR=${ROCM}/lib/cmake/rocblas/
export rocsolver_DIR=${ROCM}/lib/cmake/rocsolver/

# Set the default build prefix, i.e., build-top-level
: ${MQMC_BUILD_PREFIX:=$AOMP_REPOS_TEST/miniqmc_build}
# Set the default build directory name
: ${MQMC_BUILD_DIR:=${MQMC_BUILD_PREFIX}/build_aomp_clang}
# Path to the miniqmc source directory
: ${MQMC_SOURCE_DIR:=$AOMP_REPOS_TEST/miniqmc_src}
# how many threads should be used for building miniqmc
: ${MQMC_NUM_BUILD_PROCS:=32}
# We pin the version by default, so we have only AOMP as moving target
: ${MQMC_GIT_TAG:=15edc605}


if [ ! -d $MQMC_SOURCE_DIR ]; then
  git clone https://github.com/ye-luo/miniqmc $MQMC_SOURCE_DIR
  git checkout ${MQMC_GIT_TAG}
fi

rm -rf ${MQMC_BUILD_DIR}
# Note: We currently need the -fopenmp-assume-no-nested-parallelism to work around a call to malloc which probably should not be there.
# In the case that we disable hostservices, the application crashes when trying to call malloc.
CMAKE_PREFIX_PATH=${ROCM}/lib/cmake/ cmake -B ${MQMC_BUILD_DIR} -S ${MQMC_SOURCE_DIR} -DCMAKE_CXX_COMPILER=clang++ -DENABLE_OFFLOAD=ON -DQMC_ENABLE_ROCM=ON -DCMAKE_CXX_FLAGS='-fopenmp-assume-no-nested-parallelism -DCUDART_VERSION=10000 -DcudaMemoryTypeManaged=hipMemoryTypeManaged ' -DAMDGPU_DISABLE_HOST_DEVMEM=ON -DCMAKE_VERBOSE_MAKEFILE=ON

# Build miniqmc binaries
#cmake --build ${MQMC_BUILD_DIR}  --clean-first -j ${MQMC_NUM_BUILD_PROCS}
pushd ${MQMC_BUILD_DIR}
make clean
make --output-sync -j ${MQMC_NUM_BUILD_PROCS}
popd

echo "Running Tests"
echo "OMP_NUM_THREADS=${MQMC_OMP_NUM_THREADS} ${MQMC_BUILD_DIR}/bin/check_spo_batched_reduction -n 10"
OMP_NUM_THREADS=${MQMC_OMP_NUM_THREADS} ${MQMC_BUILD_DIR}/bin/check_spo_batched_reduction -n 10

echo ""
echo "OMP_NUM_THREADS=${MQMC_OMP_NUM_THREADS} ${MQMC_BUILD_DIR}/bin/miniqmc"
OMP_NUM_THREADS=${MQMC_OMP_NUM_THREADS} ${MQMC_BUILD_DIR}/bin/miniqmc -v

echo ""
echo "OMP_NUM_THREADS=${MQMC_OMP_NUM_THREADS} ${MQMC_BUILD_DIR}/bin/check_spo -n 10"
OMP_NUM_THREADS=${MQMC_OMP_NUM_THREADS} ${MQMC_BUILD_DIR}/bin/check_spo -n 10 -v

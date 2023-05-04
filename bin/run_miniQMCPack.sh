#! /usr/bin/env bash

# run_miniQMCPack.sh - runs several MiniQMC binaries
# The user should set at least these two variables:
# MQMC_BUILD_PREFIX       miniqmc build prefix (default: ~/miniqmc_build)
# MQMC_SOURCE_DIR         path to miniqmc sources (default: .)
#
# The user can set different variables to control execution.
# MQMC_OMP_NUM_THREADS    how many OpenMP threads should be used (default 64)
# MQMC_NUM_BUILD_PROCS    how many processes to build miniqmc (default 32)
# ROCM_INSTALL_PATH       top-level ROCm install directory (default: /opt/rocm-5.3.0)

# --- Start standard header to set AOMP environment variables ----
realpath=`realpath $0`
thisdir=`dirname $realpath`
. $thisdir/aomp_common_vars
# --- end standard header ----

# Default ROCm installation
: ${ROCM_INSTALL_PATH:=/opt/rocm-5.3.0}

# Control how many OpenMP threads are used by MiniQMCPack
: ${MQMC_OMP_NUM_THREADS:=64}

export PATH=$AOMP/bin:$PATH
#export PATH=/home/janplehr/rocm/trunk/bin:$PATH

# We export all these paths, so they will be picked-up in the CMake command.
export hsaruntime64_DIR=${ROCM_INSTALL_PATH}/lib/cmake/hsa-runtime64/
export hipblas_DIR=${ROCM_INSTALL_PATH}/lib/cmake/hipblas/
export hip_DIR=${ROCM_INSTALL_PATH}/lib/cmake/hip
export AMDDeviceLibs_DIR=${ROCM_INSTALL_PATH}/lib/cmake/AMDDeviceLibs/
export amd_comgr_DIR=${ROCM_INSTALL_PATH}/lib/cmake/amd_comgr/
export rocblas_DIR=${ROCM_INSTALL_PATH}/lib/cmake/rocblas/
export rocsolver_DIR=${ROCM_INSTALL_PATH}/lib/cmake/rocsolver/

# Set the default build prefix, i.e., build-top-level
: ${MQMCPACK_BUILD_PREFIX:=$HOME/miniqmc_build}
# Set the default build directory name
: ${MQMCPACK_BUILD_DIR:=${MQMCPACK_BUILD_PREFIX}/build_aomp_clang}
# Path to the miniqmc source directory
: ${MQMC_SOURCE_DIR:=$HOME/miniqmc_src}
# how many threads should be used for building miniqmc
: ${MQMC_NUM_BUILD_PROCS:=32}


if [ ! -d $MQMC_SOURCE_DIR ]; then
  git clone https://github.com/ye-luo/miniqmc $MQMC_SOURCE_DIR
fi

rm -rf ${MQMCPACK_BUILD_DIR}
# Note: We currently need the -fopenmp-assume-no-nested-parallelism to work around a call to malloc which probably should not be there.
# In the case that we disable hostservices, the application crashes when trying to call malloc.
CMAKE_PREFIX_PATH=${ROCM_INSTALL_PATH}/lib/cmake/ cmake -B ${MQMCPACK_BUILD_DIR} -S ${MQMC_SOURCE_DIR} -DCMAKE_CXX_COMPILER=clang++ -DENABLE_OFFLOAD=ON -DQMC_ENABLE_ROCM=ON -DCMAKE_CXX_FLAGS='-fopenmp-assume-no-nested-parallelism ' -DAMDGPU_DISABLE_HOST_DEVMEM=ON -DCMAKE_VERBOSE_MAKEFILE=ON

# Build miniqmc binaries
cmake --build ${MQMCPACK_BUILD_DIR}  --clean-first -j ${MQMC_NUM_BUILD_PROCS}

echo "Running Tests"
echo "OMP_NUM_THREADS=${MQMC_OMP_NUM_THREADS} ${MQMCPACK_BUILD_DIR}/bin/check_spo_batched_reduction -n 10"
OMP_NUM_THREADS=${MQMC_OMP_NUM_THREADS} ${MQMCPACK_BUILD_DIR}/bin/check_spo_batched_reduction -n 10

echo ""
echo "OMP_NUM_THREADS=${MQMC_OMP_NUM_THREADS} ${MQMCPACK_BUILD_DIR}/bin/miniqmc"
OMP_NUM_THREADS=${MQMC_OMP_NUM_THREADS} ${MQMCPACK_BUILD_DIR}/bin/miniqmc -v

echo ""
echo "OMP_NUM_THREADS=${MQMC_OMP_NUM_THREADS} ${MQMCPACK_BUILD_DIR}/bin/check_spo -n 10"
OMP_NUM_THREADS=${MQMC_OMP_NUM_THREADS} ${MQMCPACK_BUILD_DIR}/bin/check_spo -n 10 -v

#!/usr/bin/env bash

#
#Copyright © Advanced Micro Devices, Inc., or its affiliates.
#
#SPDX-License-Identifier:  MIT
#

# Build script for llvm-test-suite with HIP support using AOMP compiler

# --- Start standard header to set AOMP environment variables ----
realpath=$(realpath "$0")
thisdir=$(dirname "$realpath")
export AOMP_USE_CCACHE=0

# shellcheck source=aomp_common_vars
. "$thisdir/aomp_common_vars"
# --- end standard header ----

# Environment variable defaults
: "${LLVMTS_TLDIR:=$AOMP_REPOS_TEST/llvm-test-suite}"
: "${LLVMTS_SRC_DIR:=$LLVMTS_TLDIR/src}"
: "${LLVMTS_BUILD_DIR:=$LLVMTS_TLDIR/build}"
: "${LLVMTS_EXTERNAL_DIR:=$LLVMTS_TLDIR/External}"
: "${LLVMTS_LOGS_DIR:=$LLVMTS_TLDIR/logs}"
: "${LLVMTS_GPU:=$AOMP_GPU}"
: "${LLVMTS_BUILD_TYPE:=Release}"
: "${LLVMTS_TEST_TIMEOUT:=840}"
# Fallback to system ROCm if AOMP doesn't have HIP libraries
: "${ROCM:=/opt/rocm}"

# Determine clang major version for device libs path
CLANG_VERSION=$("${AOMP}/bin/clang" --version | head -1 | grep -o 'version [0-9]*' | awk '{print $2}')
if [ -z "${CLANG_VERSION}" ]; then
  echo "WARNING: Could not determine clang version, defaulting to 23"
  CLANG_VERSION=23
fi

# Set AMD device libs path for HIP compilation
# HIP tests need to find device bitcode libraries
HIP_DEVICE_LIB_PATH="${AOMP}/lib/clang/${CLANG_VERSION}/lib/amdgcn/bitcode"
export HIP_DEVICE_LIB_PATH

pushd "${AOMP_REPOS_TEST}" || exit
mkdir -p "${LLVMTS_TLDIR}" && cd "${LLVMTS_TLDIR}" || exit

# Control variables
DoConfigure='no'
DoCompile='no'
DoTest='no'
DoUpdate='no'
IsVerbose='no'

usage() {
  echo "Usage: $(basename "$0") [-j build_jobs] [-c configure] [-b build] [-t test] [-v verbose] [-u update_sources]"
  echo ""
  echo "Options:"
  echo "  -c  Run CMake configuration"
  echo "  -b  Build the test suite"
  echo "  -t  Run tests"
  echo "  -u  Update sources (git pull)"
  echo "  -j  Number of parallel build jobs (default: $AOMP_BUILD_JOBS)"
  echo "  -v  Verbose mode (set -x)"
  echo "  -h  Show this help message"
  echo ""
  echo "Environment Variables:"
  echo "  LLVMTS_TLDIR         - Top-level directory (default: \$AOMP_REPOS_TEST/llvm-test-suite)"
  echo "  LLVMTS_GPU           - Target GPU(s) (default: \$AOMP_GPU)"
  echo "  LLVMTS_BUILD_TYPE    - CMake build type (default: Release)"
  echo "  LLVMTS_TEST_TIMEOUT  - Test timeout in seconds (default: 800)"
  echo "  AOMP                 - AOMP compiler location (default: \$HOME/rocm/aomp)"
  echo "  ROCM                 - ROCm installation path (fallback, default: /opt/rocm)"
  echo "                         Only needed if AOMP doesn't contain HIP libraries"
}

# Parse options with GNU getopt, which supports long options (added later) and
# the bundled/short forms (-cbt, -j8, -j 8) the script already accepted.
if ! ParsedArgs=$(getopt -o j:cbtvhu -n "$(basename "$0")" -- "$@"); then
  usage >&2
  exit 1
fi
eval set -- "${ParsedArgs}"

while true; do
  case "$1" in
  -j)
    AOMP_BUILD_JOBS="$2"
    shift 2
    ;;
  -c)
    DoConfigure='yes'
    shift
    ;;
  -b)
    DoCompile='yes'
    shift
    ;;
  -t)
    DoTest='yes'
    shift
    ;;
  -v)
    IsVerbose='yes'
    shift
    ;;
  -u)
    DoUpdate='yes'
    shift
    ;;
  -h)
    usage
    exit 0
    ;;
  --)
    shift
    break
    ;;
  esac
done

if [ "${IsVerbose}" == "yes" ]; then
  set -x
fi

# Detect ninja vs make
TestBuildTool='make'
CmakeGenerator=""
if command -v ninja >/dev/null; then
  CmakeGenerator="-GNinja"
  TestBuildTool='ninja'
fi

# Create log directory
if [ ! -d "${LLVMTS_LOGS_DIR}" ]; then
  mkdir -p "${LLVMTS_LOGS_DIR}"
fi

# Clone or update repository
if [ ! -d "${LLVMTS_SRC_DIR}" ]; then
  echo "Cloning llvm-test-suite repository..."
  git clone https://github.com/llvm/llvm-test-suite.git "${LLVMTS_SRC_DIR}"
elif [ "${DoUpdate}" == "yes" ]; then
  echo "Updating llvm-test-suite repository..."
  cd "${LLVMTS_SRC_DIR}" || exit
  git pull
  cd "${LLVMTS_TLDIR}" || exit
fi

# Check for HIP tests in the cloned suite
if [ -d "${LLVMTS_SRC_DIR}" ] && [ ! -d "${LLVMTS_SRC_DIR}/External/HIP" ]; then
  echo "WARNING: External/HIP tests not found in llvm-test-suite"
  echo "         The test suite may not have HIP tests available"
fi

# Determine ROCm path for HIP tests
# Prefer using AOMP's root directory (which contains HIP runtime and libraries)
# over system ROCm installation. AOMP root is typically $AOMP/../.. when AOMP points
# to lib/llvm subdirectory
AOMP_ROOT_DIR=$(cd "${AOMP}/../.." && pwd)
if [ -f "${AOMP_ROOT_DIR}/lib/libamdhip64.so" ]; then
  ROCM_FOR_TESTS="${AOMP_ROOT_DIR}"
  echo "Using AOMP installation for HIP tests: ${ROCM_FOR_TESTS}"
elif [ -f "${AOMP}/../lib/libamdhip64.so" ]; then
  # Handle case where AOMP points directly to root (e.g., /opt/rocm/llvm)
  AOMP_ROOT_DIR=$(cd "${AOMP}/.." && pwd)
  ROCM_FOR_TESTS="${AOMP_ROOT_DIR}"
  echo "Using AOMP installation for HIP tests: ${ROCM_FOR_TESTS}"
else
  if [ ! -d "${ROCM}" ]; then
    echo "ERROR: AOMP installation does not contain HIP libraries and system ROCm not found at ${ROCM}"
    echo "       Please set ROCM environment variable to a valid ROCm installation"
    exit 1
  fi
  ROCM_FOR_TESTS="${ROCM}"
  echo "Using system ROCm for HIP tests: ${ROCM_FOR_TESTS}"
fi

# Export ROCm CMake directories based on selected ROCm path
export hsaruntime64_DIR=${ROCM_FOR_TESTS}/lib/cmake/hsa-runtime64/
export hipblas_DIR=${ROCM_FOR_TESTS}/lib/cmake/hipblas/
export hip_DIR=${ROCM_FOR_TESTS}/lib/cmake/hip
export AMDDeviceLibs_DIR=${ROCM_FOR_TESTS}/lib/cmake/AMDDeviceLibs/
export amd_comgr_DIR=${ROCM_FOR_TESTS}/lib/cmake/amd_comgr/

# Get ROCm version and create symlink
if [ -f "${ROCM_FOR_TESTS}/.info/version" ]; then
  ROCM_VERSION_FILE=$(cat "${ROCM_FOR_TESTS}/.info/version")
elif command -v dpkg >/dev/null && dpkg -l rocm-core >/dev/null 2>&1; then
  ROCM_VERSION_FILE=$(dpkg -l rocm-core | grep rocm-core | awk '{print $3}' | cut -d- -f1)
else
  # Default fallback version
  ROCM_VERSION_FILE="6.0.0"
fi

# Create hip subdirectory and symlink to ROCm installation
# This is a prerequisite / assumption that the LLVM test suite makes
if [ ! -d "${LLVMTS_EXTERNAL_DIR}/hip" ]; then
  mkdir -p "${LLVMTS_EXTERNAL_DIR}/hip"
fi

ROCM_LINK="${LLVMTS_EXTERNAL_DIR}/hip/rocm-${ROCM_VERSION_FILE}"
if [ ! -L "${ROCM_LINK}" ] && [ ! -d "${ROCM_LINK}" ]; then
  echo "Creating symlink: ${ROCM_LINK} -> ${ROCM_FOR_TESTS}"
  ln -sf "${ROCM_FOR_TESTS}" "${ROCM_LINK}"
fi

# Configure with CMake
if [ "${DoConfigure}" == "yes" ]; then
  echo "Configuring build with CMake..."
  rm -rf "${LLVMTS_BUILD_DIR}"
  cmake ${CmakeGenerator} \
    -B "${LLVMTS_BUILD_DIR}" \
    -S "${LLVMTS_SRC_DIR}" \
    -DTEST_SUITE_SUBDIRS=External \
    -DTEST_SUITE_EXTERNALS_DIR="${LLVMTS_EXTERNAL_DIR}" \
    -DTEST_SUITE_COLLECT_CODE_SIZE=OFF \
    -DTEST_SUITE_COLLECT_COMPILE_TIME=OFF \
    -DTEST_SUITE_LIT="${AOMP}/bin/llvm-lit" \
    -DCMAKE_STRIP="" \
    -DAMDGPU_ARCHS="${LLVMTS_GPU}" \
    -DHIP_EXPORT_XUNIT_XML=ON \
    -DENABLE_HIP_CATCH_TESTS=ON \
    -DCMAKE_BUILD_TYPE="${LLVMTS_BUILD_TYPE}" \
    -DCMAKE_C_COMPILER="${AOMP}/bin/clang" \
    -DCMAKE_CXX_COMPILER="${AOMP}/bin/clang++"
fi

# Build
if [ "${DoCompile}" == "yes" ]; then
  echo "Building llvm-test-suite..."
  cmake --build "${LLVMTS_BUILD_DIR}" --parallel -j "${AOMP_BUILD_JOBS}"
fi

# Run tests
if [ "${DoTest}" == "yes" ]; then
  echo "Running HIP tests (timeout: ${LLVMTS_TEST_TIMEOUT}s)..."
  cd "${LLVMTS_BUILD_DIR}" || exit

  echo "Log in ${LLVMTS_LOGS_DIR}/test-output.log"

  for TestTarget in check-hip-simple check-hip-catch; do
  # Use timeout to prevent tests from hanging
  # -k 30: Send SIGKILL after 30 seconds if SIGTERM doesn't terminate the process
  # This ensures even completely hung processes are killed
  if command -v timeout >/dev/null; then
    timeout -k 30 "${LLVMTS_TEST_TIMEOUT}" "${TestBuildTool}" "${TestTarget}" 2>&1 | tee -a "${LLVMTS_LOGS_DIR}/test-output.log"
    test_exit_code="${PIPESTATUS[0]}"
    if [ "${test_exit_code}" -eq 124 ]; then
      echo "WARNING: Tests timed out after ${LLVMTS_TEST_TIMEOUT} seconds"
    elif [ "${test_exit_code}" -eq 137 ]; then
      echo "WARNING: Tests were forcefully killed (SIGKILL) after timeout grace period"
    fi
  else
    echo "WARNING: timeout command not found. Not running tests"
    popd || exit 1
    exit 1
  fi
  done
fi

popd || exit

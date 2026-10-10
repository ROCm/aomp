#!/bin/bash
#
# run_mglet.sh:
#   Build MGLET-base with the AOMP OpenMP-offload toolchain (amdclang,
#   amdclang++, amdflang) and run the upstream CTest suite.
#
# The MGLET-base sources are cloned into $AOMP_REPOS_TEST by clone_test.sh
# (see manifests/test_<version>.xml). Source repo:
#   https://github.com/kmturbulenz/mglet-base
#
# MGLET-base requires MPI and parallel HDF5. Build the AOMP supplemental
# dependencies first with:
#   ./build_supp.sh rocmopenmpi hdf5-parallel
#
# Usage:
#   ./run_mglet.sh          configure + build + run tests
#   ./run_mglet.sh nocmake  skip CMake configure, rebuild + test
#   ./run_mglet.sh rerun    skip build, just re-run tests
#

# --- Start standard header to set AOMP environment variables ----
realpath=$(realpath "$0")
thisdir=$(dirname "$realpath")
export AOMP_USE_CCACHE=0

# shellcheck disable=SC1091
. "$thisdir"/aomp_common_vars
# --- end standard header ----

# Offload kernel tracing is very noisy across a full test-suite; default off.
export LIBOMPTARGET_KERNEL_TRACE=${LIBOMPTARGET_KERNEL_TRACE:-0}

# Setup AOMP variables
AOMP=${AOMP:-$HOME/rocm/srock/llvm}

# Use function to set and test AOMP_GPU
setaompgpu

# --- MGLET specific variables (all overridable from the environment) --------

# Directory name the MGLET-base repo is cloned into (matches manifest path=).
AOMP_MGLET_REPO_NAME=${AOMP_MGLET_REPO_NAME:-mglet-base}
MGLET_REPO=${MGLET_REPO:-$AOMP_REPOS_TEST/$AOMP_MGLET_REPO_NAME}
MGLET_BUILD=${MGLET_BUILD:-$MGLET_REPO/build-aomp-$AOMP_GPU}

# GPU arch passed to MGLET CMake offload flags. Defaults to detected AOMP_GPU.
GPU_ARCH=${GPU_ARCH:-$AOMP_GPU}

# Supplemental components built by build_supp.sh.
OPENMPI_INSTALL=${OPENMPI_INSTALL:-$AOMP_SUPP/rocmopenmpi}
HDF5_INSTALL=${HDF5_INSTALL:-$AOMP_SUPP/hdf5-parallel}

# ROCm root used to find HIP/OpenMP runtime support. For an AOMP install under
# a ROCm tree, $AOMP is usually <rocm>/lib/llvm.
ROCM_PATH=${ROCM_PATH:-"$(realpath -m "$(realpath -m "$AOMP")/../..")"}

# Make the AOMP compilers and supplemental MPI/HDF5 visible to CMake and tests.
export PATH=$OPENMPI_INSTALL/bin:$AOMP/bin:$ROCM_PATH/bin:$ROCM_PATH/llvm/bin:$PATH
export LD_LIBRARY_PATH=$OPENMPI_INSTALL/lib:$HDF5_INSTALL/lib:$AOMP/lib:$ROCM_PATH/lib:$ROCM_PATH/lib64:$LD_LIBRARY_PATH
export LIBRARY_PATH=$HDF5_INSTALL/lib:$AOMP/lib:$ROCM_PATH/lib:$ROCM_PATH/lib64:$LIBRARY_PATH

# OpenMPI wrappers may have been built with another compiler. Force wrapper
# compiler identity so CMake and MPI use the AOMP toolchain consistently.
export OMPI_CC=${OMPI_CC:-amdclang}
export OMPI_CXX=${OMPI_CXX:-amdclang++}
export OMPI_FC=${OMPI_FC:-amdflang}

# Runtime environment.
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-1}
export HSA_ENABLE_IPC_MODE_LEGACY=${HSA_ENABLE_IPC_MODE_LEGACY:-1}
export HSA_ENABLE_SDMA=${HSA_ENABLE_SDMA:-1}
export OMPX_FORCE_SYNC_REGIONS=${OMPX_FORCE_SYNC_REGIONS:-1}

AOMP_CMAKE=${AOMP_CMAKE:-cmake}
AOMP_CTEST=${AOMP_CTEST:-ctest}
MGLET_CTEST_ARGS=${MGLET_CTEST_ARGS:-"--output-on-failure"}

if [ ! -d "$MGLET_REPO" ]; then
  echo "ERROR: MGLET-base sources not found at $MGLET_REPO"
  echo "       Run clone_test.sh first (needs a manifest entry for MGLET-base)."
  exit 1
fi

if [ ! -x "$OPENMPI_INSTALL/bin/mpirun" ]; then
  echo "ERROR: OpenMPI not found at $OPENMPI_INSTALL"
  echo "       Build AOMP supplemental ROCm-aware OpenMPI first with build_supp.sh rocmopenmpi,"
  echo "       or set OPENMPI_INSTALL."
  exit 1
fi

if [ ! -d "$HDF5_INSTALL" ]; then
  echo "ERROR: HDF5 not found at $HDF5_INSTALL"
  echo "       Build AOMP supplemental parallel HDF5 first with build_supp.sh hdf5-parallel,"
  echo "       or set HDF5_INSTALL."
  exit 1
fi

ulimit -s unlimited

run_mglet_tests() {
  cd "$MGLET_BUILD" || exit 1

  echo "=== MGLET-base test configuration ==="
  echo "Repo:       $MGLET_REPO"
  echo "Build:      $MGLET_BUILD"
  echo "GPU arch:   $GPU_ARCH"
  echo "OpenMPI:    $OPENMPI_INSTALL"
  echo "HDF5:       $HDF5_INSTALL"
  echo "CTest args: $MGLET_CTEST_ARGS"
  echo "=== End MGLET-base test configuration ==="

  # shellcheck disable=SC2086
  $AOMP_CTEST --test-dir "$MGLET_BUILD/tests" $MGLET_CTEST_ARGS
  ret=$?

  if [ "$ret" -ne 0 ]; then
    echo "mglet-base" >> "$MGLET_REPO"/failing-tests.txt
  else
    echo "mglet-base" >> "$MGLET_REPO"/passing-tests.txt
  fi
  return "$ret"
}

if [ "$1" == "rerun" ]; then
  run_mglet_tests
  exit $?
fi

cd "$MGLET_REPO" || exit 1
rm -f make-fail.txt failing-tests.txt passing-tests.txt

if [ "$1" != "nocmake" ]; then
  echo "Configuring MGLET-base with AOMP OpenMP offload for $GPU_ARCH"
  rm -rf "$MGLET_BUILD"
  mkdir -p "$MGLET_BUILD"

  cmake_prefix_path="$OPENMPI_INSTALL;$HDF5_INSTALL;$ROCM_PATH"
  cmake_args=(
    -S "$MGLET_REPO"
    -B "$MGLET_BUILD"
    -DCMAKE_BUILD_TYPE=Release
    -DCMAKE_C_COMPILER=amdclang
    -DCMAKE_CXX_COMPILER=amdclang++
    -DCMAKE_Fortran_COMPILER=amdflang
    -DMPI_C_COMPILER="$OPENMPI_INSTALL/bin/mpicc"
    -DMPI_CXX_COMPILER="$OPENMPI_INSTALL/bin/mpicxx"
    -DMPI_Fortran_COMPILER="$OPENMPI_INSTALL/bin/mpifort"
    -DHDF5_ROOT="$HDF5_INSTALL"
    -DCMAKE_PREFIX_PATH="$cmake_prefix_path"
    -DMGLET_C_FLAGS=
    -DMGLET_CXX_FLAGS=
    "-DMGLET_Fortran_FLAGS=-fimplicit-none;-Wno-assumed-type-size-dummy"
    "-DMGLET_C_FLAGS_RELEASE=-O3;-g"
    "-DMGLET_CXX_FLAGS_RELEASE=-O3;-g"
    "-DMGLET_Fortran_FLAGS_RELEASE=-O3;-g"
    -DMGLET_OFFLOAD=ON
    -DMGLET_WORKAROUNDS=ON
    -DMGLET_OFFLOAD_COMPILE_FLAGS=-fopenmp-version=52
    -DMGLET_OFFLOAD_ARCH_FLAGS=--offload-arch="$GPU_ARCH"
  )

  echo "$AOMP_CMAKE ${cmake_args[*]}"
  "$AOMP_CMAKE" "${cmake_args[@]}"
  ret=$?
  if [ "$ret" -ne 0 ]; then
    echo "mglet-base-configure" >> make-fail.txt
    exit 1
  fi
fi

echo "Building MGLET-base"
"$AOMP_CMAKE" --build "$MGLET_BUILD" -j"$AOMP_JOB_THREADS"
ret=$?
if [ "$ret" -ne 0 ]; then
  echo "mglet-base" >> make-fail.txt
  exit 1
fi

run_mglet_tests
exit $?

#!/bin/bash
#
# run_ecrad.sh:
#   Build ecRad (ECMWF atmospheric radiation scheme) with the AOMP
#   OpenMP-offload toolchain (amdclang, amdflang), run the upstream CTest
#   suite on the GPU, then run the benchmark case and compare its output
#   against a reference.
#
# The ecRad sources are cloned into $AOMP_REPOS_TEST by clone_test.sh
# (see manifests/test_<version>.xml). Source repo:
#   https://github.com/ecmwf-ifs/ecrad     (branch master-omp)
#
# ecRad requires serial NetCDF-Fortran, ecbuild and fiat. Build the AOMP
# supplemental dependencies first with:
#   ./build_supp.sh netcdf-fortran fiat
#
# By default the CTest suite includes the bit-identity checks, which compare
# the GPU drivers against the standalone driver. CMake creates a Python venv.
# Set ECRAD_BITIDENTITY=OFF to only check that the drivers run.
#
# The benchmark needs a data directory (not in the repo) at $ECRAD_DATA_DIR,
# default $AOMP_REPOS_TEST/ecrad-data, with:
#   rad4NN_inputs_2020010100_0c0.nc            input
#   ecrad_ifs_blocked_dp_B200-master-acc.nc    reference output
#   config.nam                                 namelist
# The comparison uses the nccmp tool of the ecRad build (needs
# ECRAD_BITIDENTITY=ON). Set ECRAD_BENCHMARK=OFF to skip the benchmark.
#
# Usage:
#   ./run_ecrad.sh          configure + build ecRad + run tests + benchmark
#   ./run_ecrad.sh nocmake  skip CMake configure, rebuild + tests + benchmark
#   ./run_ecrad.sh rerun    skip build, just re-run tests + benchmark
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

# Fail instead of silently falling back to the host if offloading does not work.
export OMP_TARGET_OFFLOAD=${OMP_TARGET_OFFLOAD:-MANDATORY}

# Setup AOMP variables
AOMP=${AOMP:-$HOME/rocm/srock/llvm}

# Use function to set and test AOMP_GPU
setaompgpu

# --- ecRad specific variables (all overridable from the environment) -------

# Directory name the ecRad repo is cloned into (matches manifest path=).
AOMP_ECRAD_REPO_NAME=${AOMP_ECRAD_REPO_NAME:-ecrad}
ECRAD_REPO=${ECRAD_REPO:-$AOMP_REPOS_TEST/$AOMP_ECRAD_REPO_NAME}
ECRAD_BUILD=${ECRAD_BUILD:-$ECRAD_REPO/build-aomp-$AOMP_GPU}

# GPU arch passed to the ecRad offload flags. Defaults to detected AOMP_GPU.
GPU_ARCH=${GPU_ARCH:-$AOMP_GPU}

# Supplemental components built by build_supp.sh.
HDF5_INSTALL=${HDF5_INSTALL:-$AOMP_SUPP/hdf5}
NETCDF_C_INSTALL=${NETCDF_C_INSTALL:-$AOMP_SUPP/netcdf-c}
NETCDF_F_INSTALL=${NETCDF_F_INSTALL:-$AOMP_SUPP/netcdf-fortran}
ECBUILD_INSTALL=${ECBUILD_INSTALL:-$AOMP_SUPP/ecbuild}
FIAT_INSTALL=${FIAT_INSTALL:-$AOMP_SUPP/fiat}

# ROCm root used to find HIP/OpenMP runtime support. For an AOMP install under
# a ROCm tree, $AOMP is usually <rocm>/lib/llvm.
ROCM_PATH=${ROCM_PATH:-"$(realpath -m "$(realpath -m "$AOMP")/../..")"}

# Compiler flags. Same as ecRad AMD GPU build, with offload arch taken from GPU_ARCH.
ECRAD_Fortran_FLAGS=${ECRAD_Fortran_FLAGS:-"-march=native -fdefault-real-8 -O3 --offload-arch=$GPU_ARCH -ffast-real-mod -fopenmp"}
ECRAD_LINKER_FLAGS=${ECRAD_LINKER_FLAGS:-"--offload-arch=$GPU_ARCH -fopenmp -march=native -O3"}

# Bit-identity checks of the test outputs (needs Python venv).
ECRAD_BITIDENTITY=${ECRAD_BITIDENTITY:-ON}

# Benchmark: data directory
ECRAD_BENCHMARK=${ECRAD_BENCHMARK:-ON}
ECRAD_DATA_DIR=${ECRAD_DATA_DIR:-$AOMP_REPOS_TEST/ecrad-data}
ECRAD_BENCH_DIR=${ECRAD_BENCH_DIR:-$ECRAD_BUILD/benchmark}
ECRAD_BENCH_INPUT=${ECRAD_BENCH_INPUT:-rad4NN_inputs_2020010100_0c0.nc}
ECRAD_BENCH_REFERENCE=${ECRAD_BENCH_REFERENCE:-ecrad_ifs_blocked_dp_B200-master-acc.nc}
ECRAD_NBLOCKSIZE=${ECRAD_NBLOCKSIZE:-8480}
ECRAD_THRESHOLD=${ECRAD_THRESHOLD:-0.001}

# Offload runtime settings used by the ecRad app team for the benchmark.
export LIBOMPTARGET_AMDGPU_MAX_ASYNC_COPY_BYTES=${LIBOMPTARGET_AMDGPU_MAX_ASYNC_COPY_BYTES:-0}
export LIBOMPTARGET_AMDGPU_TEAMS_PER_CU=${LIBOMPTARGET_AMDGPU_TEAMS_PER_CU:-4}

# Make the AOMP compilers and supplemental libraries visible to CMake and tests.
export PATH=$AOMP/bin:$ROCM_PATH/bin:$ROCM_PATH/llvm/bin:$PATH
export LD_LIBRARY_PATH=$NETCDF_F_INSTALL/lib:$NETCDF_C_INSTALL/lib:$HDF5_INSTALL/lib:$AOMP/lib:$ROCM_PATH/lib:$ROCM_PATH/lib64:$LD_LIBRARY_PATH
export LIBRARY_PATH=$AOMP/lib:$ROCM_PATH/lib:$ROCM_PATH/lib64:$LIBRARY_PATH

AOMP_CMAKE=${AOMP_CMAKE:-cmake}
AOMP_CTEST=${AOMP_CTEST:-"$(dirname "$AOMP_CMAKE")/ctest"}
ECRAD_CTEST_ARGS=${ECRAD_CTEST_ARGS:-"--output-on-failure"}

if [ ! -d "$ECRAD_REPO" ]; then
  echo "ERROR: ecRad sources not found at $ECRAD_REPO"
  echo "       Run clone_test.sh first (needs a manifest entry for ecrad)."
  exit 1
fi

if [ ! -d "$NETCDF_F_INSTALL" ]; then
  echo "ERROR: NetCDF-Fortran not found at $NETCDF_F_INSTALL"
  echo "       Build AOMP supplemental NetCDF first with build_supp.sh netcdf-c netcdf-fortran,"
  echo "       or set NETCDF_F_INSTALL and NETCDF_C_INSTALL."
  exit 1
fi

if [ ! -d "$ECBUILD_INSTALL" ] || [ ! -d "$FIAT_INSTALL" ]; then
  echo "ERROR: ecbuild or fiat not found at $ECBUILD_INSTALL, $FIAT_INSTALL"
  echo "       Build AOMP supplemental fiat first with build_supp.sh fiat,"
  echo "       or set ECBUILD_INSTALL and FIAT_INSTALL."
  exit 1
fi

ulimit -s unlimited

run_ecrad_ctest() {
  cd "$ECRAD_BUILD" || exit 1

  echo "=== ecRad test configuration ==="
  echo "Repo:         $ECRAD_REPO"
  echo "Build:        $ECRAD_BUILD"
  echo "fiat:         $FIAT_INSTALL"
  echo "GPU arch:     $GPU_ARCH"
  echo "NetCDF:       $NETCDF_F_INSTALL"
  echo "Bit-identity: $ECRAD_BITIDENTITY"
  echo "CTest args:   $ECRAD_CTEST_ARGS"
  echo "=== End ecRad test configuration ==="

  # shellcheck disable=SC2086
  "$AOMP_CTEST" --test-dir "$ECRAD_BUILD" $ECRAD_CTEST_ARGS
  ret=$?

  if [ "$ret" -ne 0 ]; then
    echo "ecrad" >> "$ECRAD_REPO"/failing-tests.txt
  else
    echo "ecrad" >> "$ECRAD_REPO"/passing-tests.txt
  fi
  return "$ret"
}

# Run ecrad_ifs_blocked_dp driver on benchmark input and compare output 
# against reference. Timings are printed (but not checked).
run_ecrad_benchmark() {
  local nccmp=$ECRAD_BUILD/test/common/nccmp
  local exe=$ECRAD_BUILD/bin/ecrad_ifs_blocked_dp
  local out=$ECRAD_BENCH_DIR/out-ecrad_ifs_blocked_dp-np$ECRAD_NBLOCKSIZE.nc
  local file

  echo "=== ecRad benchmark configuration ==="
  echo "Data:       $ECRAD_DATA_DIR"
  echo "Work dir:   $ECRAD_BENCH_DIR"
  echo "Blocksize:  $ECRAD_NBLOCKSIZE"
  echo "Reference:  $ECRAD_BENCH_REFERENCE"
  echo "Threshold:  $ECRAD_THRESHOLD"
  echo "=== End ecRad benchmark configuration ==="

  for file in "$ECRAD_BENCH_INPUT" "$ECRAD_BENCH_REFERENCE" config.nam; do
    if [ ! -f "$ECRAD_DATA_DIR/$file" ]; then
      echo "ERROR: ecRad benchmark data $file not found in $ECRAD_DATA_DIR"
      echo "       Set ECRAD_DATA_DIR, or ECRAD_BENCHMARK=OFF to skip the benchmark."
      echo "ecrad-benchmark" >> "$ECRAD_REPO"/failing-tests.txt
      return 1
    fi
  done
  if [ ! -x "$nccmp" ]; then
    echo "ERROR: $nccmp not found, the benchmark comparison needs ECRAD_BITIDENTITY=ON."
    echo "ecrad-benchmark" >> "$ECRAD_REPO"/failing-tests.txt
    return 1
  fi

  rm -rf "$ECRAD_BENCH_DIR"
  mkdir -p "$ECRAD_BENCH_DIR"
  cd "$ECRAD_BENCH_DIR" || exit 1

  # The namelist reads the lookup tables from ./data.
  ln -s "$ECRAD_REPO/data" data
  sed -E "s/^[[:space:]]*nblocksize[[:space:]]*=[[:space:]]*[0-9]+,/nblocksize=$ECRAD_NBLOCKSIZE,/" \
    "$ECRAD_DATA_DIR/config.nam" > config.nam

  "$exe" config.nam "$ECRAD_DATA_DIR/$ECRAD_BENCH_INPUT" "$out" 2>&1 | tee benchmark.log
  ret=${PIPESTATUS[0]}
  if [ "$ret" -eq 0 ]; then
    echo "=== ecRad benchmark timings ==="
    grep "time elapsed" benchmark.log
    "$nccmp" --longwave-threshold="$ECRAD_THRESHOLD" --shortwave-threshold="$ECRAD_THRESHOLD" \
      "$out" "$ECRAD_DATA_DIR/$ECRAD_BENCH_REFERENCE"
    ret=$?
  fi

  if [ "$ret" -ne 0 ]; then
    echo "ecRad benchmark FAILED (run or comparison against $ECRAD_BENCH_REFERENCE)"
    echo "ecrad-benchmark" >> "$ECRAD_REPO"/failing-tests.txt
  else
    echo "ecRad benchmark passed, output matches $ECRAD_BENCH_REFERENCE within $ECRAD_THRESHOLD W/m^2"
    echo "ecrad-benchmark" >> "$ECRAD_REPO"/passing-tests.txt
  fi
  return "$ret"
}

run_ecrad_tests() {
  local status=0
  run_ecrad_ctest || status=1
  if [ "$ECRAD_BENCHMARK" == "ON" ]; then
    run_ecrad_benchmark || status=1
  fi
  return "$status"
}

if [ "$1" == "rerun" ]; then
  cd "$ECRAD_REPO" || exit 1
  rm -f failing-tests.txt passing-tests.txt
  run_ecrad_tests
  exit $?
fi

cd "$ECRAD_REPO" || exit 1
rm -f make-fail.txt failing-tests.txt passing-tests.txt

if [ "$1" != "nocmake" ]; then
  echo "Configuring ecRad with AOMP OpenMP offload for $GPU_ARCH"
  rm -rf "$ECRAD_BUILD"
  mkdir -p "$ECRAD_BUILD"

  cmake_prefix_path="$NETCDF_F_INSTALL;$NETCDF_C_INSTALL;$HDF5_INSTALL;$ROCM_PATH"
  cmake_args=(
    -S "$ECRAD_REPO"
    -B "$ECRAD_BUILD"
    -Decbuild_ROOT="$ECBUILD_INSTALL"
    -Dfiat_ROOT="$FIAT_INSTALL"
    -DCMAKE_PREFIX_PATH="$cmake_prefix_path"
    -DCMAKE_BUILD_TYPE=Release
    -DCMAKE_C_COMPILER=amdclang
    -DCMAKE_Fortran_COMPILER=amdflang
    -DECBUILD_Fortran_FLAGS="$ECRAD_Fortran_FLAGS"
    -DCMAKE_EXE_LINKER_FLAGS_INIT="$ECRAD_LINKER_FLAGS"
    -DBUILD_SHARED_LIBS=OFF
    -DENABLE_OMP=ON
    -DENABLE_GPU=ON
    -DENABLE_CPU=OFF
    -DENABLE_ACC=OFF
    -DENABLE_NVTX=OFF
    -DENABLE_ROCTX=OFF
    -DENABLE_SINGLE_PRECISION=ON
    -DENABLE_DOUBLE_PRECISION=ON
    -DENABLE_BITIDENTITY_TESTING="$ECRAD_BITIDENTITY"
  )

  echo "$AOMP_CMAKE ${cmake_args[*]}"
  "$AOMP_CMAKE" "${cmake_args[@]}"
  ret=$?
  if [ "$ret" -ne 0 ]; then
    echo "ecrad-configure" >> make-fail.txt
    exit 1
  fi
fi

echo "Building ecRad"
"$AOMP_CMAKE" --build "$ECRAD_BUILD" -j"$AOMP_JOB_THREADS"
ret=$?
if [ "$ret" -ne 0 ]; then
  echo "ecrad" >> make-fail.txt
  exit 1
fi

run_ecrad_tests
exit $?

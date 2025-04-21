#!/usr/bin/env bash

CKRepoURL='https://github.com/ROCm/composable_kernel.git'
CKRepoBranchName='develop'
CKBenchmarkRepoBranchName='main'

# We grab the total system memory and assume a requirement of 10GB per process
# when building CK. This is likely a bit conservative.
CKBuildParallelism=$(free -g | grep Mem | awk '{print int($2/10)}')

realpath=$(realpath $0)
thisdir=$(dirname $realpath)

. $thisdir/aomp_common_vars

export PATH=$AOMP/bin:$PATH

function printHelp {
  echo "Usage: run_composable-kernels.sh"
  echo "  -h: Show this help message"
  echo "  -i: Install the (incremental) CK build"
  echo "  -r: Rebuild the CK repo"
  echo "  -u: Update the CK repo"
  echo "  -b: Update the CK benchmarks repo"
  exit 0
}

# Some tests may require an installed instance of CK.
ShouldInstallCK='no'
# For some situations during testing it may not be desired to rebuild the CK repo.
ShouldRebuildCK='no'
# While doing perf / other compiler work, keeping CK fix is useful.
ShouldUpdateCKRepo='no'

# CK Benchmarks is priate, maybe do not want to update it.
ShouldUpdateCKBenchmarks='no'

while getopts "hirub" opt; do
  case $opt in
  h)
    printHelp
    ;;
  i)
    # Install the CK build
    ShouldInstallCK='yes'
    ;;
  r)
    # Rebuild the CK repo
    ShouldRebuildCK='yes'
    ;;
  u)
    # Update the CK repo
    ShouldUpdateCKRepo='yes'
    ;;
  b)
    # Update the CK benchmarks repo
    ShouldUpdateCKBenchmarks='yes'
    ;;
  *)
    echo "Unknown option: -$opt"
    exit 1
    ;;
  esac
done

# Set the default build prefix, i.e., build-top-level
: ${CK_TOP:=$AOMP_REPOS_TEST/composable-kernels}
: ${CK_REPO:=$CK_TOP/ck-src}
: ${CK_BUILD:=$CK_TOP/ck-build}
: ${CK_BENCHMARK_REPO:=$CK_TOP/ck-benchmark}
# Move this to its own place, to avoid potential permission conflicts with certain setups.
: ${CK_BENCHMARK_RESULT:=$CK_TOP/ck-benchmark-result}
: ${CK_INSTALL:=$CK_TOP/ck-install}

# Get some info on the system
: ${ROCM_PATH:=/opt/rocm}
: ${CK_GPU_TARGETS:=$(amdgpu-arch)}

# Check if user overrode number of parallel build jobs
if [ ! -z ${CK_BUILD_PARALLELISM} ]; then
  CKBuildParallelism=${CK_BUILD_PARALLELISM}
fi

if [ ! -d ${CK_TOP} ]; then
  mkdir -p ${CK_TOP} || exit 1
fi

if [ ! -d ${CK_REPO} ]; then
  git clone ${CKRepoURL} ${CK_REPO}
elif [ "${ShouldUpdateCKRepo}" == 'yes' ]; then
  pushd ${CK_REPO} || exit 1
  git reset --hard origin/${CKRepoBranchName}
  git pull
  # TODO: Write current SHA to somewhere such that it is known which SHA
  #       was tested in this nightly run.
  popd
fi

# TODO Fix / Finalize the cmake command
CKCmakeCmd="cmake -GNinja -B ${CK_BUILD} -S ${CK_REPO} -DCMAKE_PREFIX_PATH=${ROCM_PATH} -DCMAKE_INSTALL_PREFIX=${CK_INSTALL} "
CKCmakeCmd+="-DCMAKE_CXX_COMPILER=${AOMP}/bin/clang++ -DCMAKE_HIP_COMPILER=${AOMP}/bin/clang++ "
CKCmakeCmd+="-DCMAKE_CXX_COMPILER_LAUNCHER=ccache "
CKCmakeCmd+="-DCMAKE_BUILD_TYPE=Release -DGPU_TARGETS=${CK_GPU_TARGETS}"

if [ "${ShouldRebuildCK}" == 'yes' ]; then
  echo "Rebuilding the CK repo"
  rm -rf ${CK_BUILD} || exit 1

  echo "CMake Config Command:"
  echo "${CKCmakeCmd}"

  ${CKCmakeCmd}
  if [ $? -ne 0 ]; then
    exit 1
  fi

  pushd ${CK_BUILD} || exit 1

  time ninja -j ${CKBuildParallelism}
  if [ $? -ne 0 ]; then
    exit 1
  fi

  popd
fi

if [ "${ShouldInstallCK}" == 'yes' ]; then
  pushd ${CK_BUILD} || exit 1

  time ninja -j ${CKBuildParallelism}
  if [ $? -ne 0 ]; then
    exit 1
  fi

  # TODO: Check parallelism. This may use all available threads.
  time ninja install
  if [ $? -ne 0 ]; then
    exit 1
  fi

  popd
fi

# The CK benchmarks repo appears to be private (for the time being).

if [ ! -d ${CK_BENCHMARK_REPO} ]; then
  echo "CK Benchmarks repo not found. This is a private repo."
  echo "Please clone with your preferred method into ${CK_BENCHMARK_REPO}"
  exit 1
elif [ "${ShouldUpdateCKBenchmarks}" == 'yes' ]; then
  pushd ${CK_BENCHMARK_REPO} || exit 1
  git reset --hard origin/${CKBenchmarkRepoBranchName}
  git pull
  # TODO: Dump SHA somewhere
  popd
fi

if [ ! -d ${CK_BENCHMARK_RESULT} ]; then
  mkdir -p ${CK_BENCHMARK_RESULT} || exit 1
fi

# This is the command. It requires the envar CK_PROFILER_DIR to be set to the directory
# in the CK build tree that contains the CkProfiler binary.
CKBenchmarkTest='../benchmarks/gemm/fa1.yaml'
CKBenchmarkName=$(basename ${CKBenchmarkTest})
CKBenchmarkResultOutput="${CK_BENCHMARK_RESULT}/${CKBenchmarkName}.output"
CKBenchmarkBackend='ck'
CKBenchmarkCmd="./run_gemm.py ${CKBenchmarkBackend} ${CKBenchmarkTest} --output ${CKBenchmarkResultOutput}"
CKBenchmarkEnvAdditions="export CK_PROFILER_DIR=${CK_BUILD}/bin"

pushd ${CK_BENCHMARK_REPO}/scripts || exit 1

echo "Benchmark Command: ${CKBenchmarkEnvAdditions} ; ${CKBenchmarkCmd}"
${CKBenchmarkEnvAdditions}
${CKBenchmarkCmd}

popd

echo "Benchmark Output File: ${CKBenchmarkResultOutput}"

#!/usr/bin/env bash

CKRepoURL='https://github.com/ROCm/composable_kernel.git'
CKRepoBranchName='develop'
CKBenchmarkRepoBranchName='main'
# XXX: CK requires quite a bit of memory when compiling.
#      Be aware! At least 6GB/core should be given
CKBuildParallelism='64'

realpath=$(realpath $0)
thisdir=$(dirname $realpath)

. $thisdir/aomp_common_vars

export PATH=$AOMP/bin:$PATH

# Set the default build prefix, i.e., build-top-level
: ${CK_TOP:=$AOMP_REPOS_TEST/composable-kernels}
: ${CK_REPO:=$CK_TOP/ck-src}
: ${CK_BUILD:=$CK_TOP/ck-build}
: ${CK_BENCHMARK_REPO:=$CK_TOP/ck-benchmark}

# Get some info on the system
: ${ROCM_PATH:=/opt/rocm}
: ${CK_GPU_TARGETS:=$(amdgpu-arch)}

if [ ! -d ${CK_TOP} ]; then
  mkdir -p ${CK_TOP} || exit 1
fi

if [ ! -d ${CK_REPO} ]; then
  git clone ${CKRepoURL} ${CK_REPO}
else
  pushd ${CK_REPO} || exit 1
  git reset --hard origin/${CKRepoBranchName}
  git pull
  # TODO: Write current SHA to somewhere such that it is known which SHA
  #       was tested in this nightly run.
  popd
fi

if [ ! -d ${CK_BENCHMARK_REPO} ]; then
  git clone ${CKBenchmarkRepoURL} ${CK_BENCHMARK_REPO}
else
  pushd ${CK_BENCHMARK_REPO} || exit 1
  git reset --hard origin/${CKBenchmarkRepoBranchName}
  git pull
  # TODO: Dump SHA somewhere
  popd
fi

# TODO Fix / Finalize the cmake command
CKCmakeCmd="cmake -GNinja -B ${CK_BUILD} -S ${CK_REPO} -DCMAKE_PREFIX_PATH=${ROCM_PATH} "
CKCMakeCmd+="-DCMAKE_CXX_COMPILER=${AOMP}/bin/clang++ -DCMAKE_HIP_COMPILER=${AOMP}/bin/clang++ "
CKCMakeCmd+="-DCMAKE_CXX_COMPILER_LAUNCHER=ccache "
CKCmakeCmd+="-DCMAKE_BUILD_TYPE=Release -DGPU_TARGETS=${CK_GPU_TARGETS}"

# TODO: Run the Cmake command
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

# The CK benchmarks repo appears to be private (for the time being).
# This is the command. It requires the envar CK_PROFILER_DIR to be set to the directory
# in the CK build tree that contains the CkProfiler binary.
CKBenchmarkTest='../benchmarks/gemm/fa1.yaml'
CKBenchmarkBackend='ck'
CKBenchmarkCmd="./run_gemm.py ${CKBenchmarkBackend} ${CKBenchmarkTest} --output ${CKBenchmarkTest}.output"
CKBenchmarkEnvAdditions="export CK_PROFILER_DIR=${CK_BUILD}/bin"

pushd ${CK_BENCHMARK_REPO}/scripts || exit 1

echo "Benchmark Command: ${CKBenchmarkEnvAdditions} ; ${CKBenchmarkCmd}"
${CKBenchmarkEnvAdditions}
${CKBenchmarkCmd}

popd

echo "Benchmark Output File: ${CKBenchmarkTest}.output"

#!/usr/bin/env bash

#
#Copyright © Advanced Micro Devices, Inc., or its affiliates.
#
#SPDX-License-Identifier:  MIT
#

# Build script for LLaMA with HIP support using AOMP compiler

. aomp_common_vars

: AOMP_GPU=${AOMP_GPU:=gfx90a}
: ${LLAMA_GPU:=$AOMP_GPU}

: ${LLAMA_TLDIR:=$AOMP_REPOS_TEST/llama}
: ${LLAMA_BUILD_DIR:=$LLAMA_TLDIR/build}
: ${LLAMA_SRC_DIR:=$LLAMA_TLDIR/src}
: ${LLAMA_BUILD_MODE:=Release}
: ${LLAMA_TESTS_LOG_LOCATION:=$LLAMA_TLDIR/logs}

# Model to use in benchmarks (default is a smaller model)
: ${LLAMA_BENCH_HF_ID:="ggml-org/gemma-3-1b-it-GGUF"}
: ${LLAMA_CACHE:="$HOME/.cache/llama.cpp"}

pushd ${AOMP_REPOS_TEST}
mkdir -p ${LLAMA_TLDIR} && cd ${LLAMA_TLDIR}

# Run CMake configuration
DoConfigure='no'

# Run build command
DoCompile='no'

# Run ctest
DoCTest='no'

# Run benchmark (llama-bench)
DoBenchmark='no'

# Update llama sources
DoUpdate='no'

IsVerbose='no'

while getopts "j:cbtveu" opt; do
  case ${opt} in
  j) AOMP_BUILD_JOBS=${OPTARG} ;;
  c) DoConfigure='yes' ;;
  b) DoCompile='yes' ;;
  t) DoCTest='yes' ;;
  v) IsVerbose='yes' ;;
  e) DoBenchmark='yes' ;;
  u) DoUpdate='yes' ;;
  \?)
    echo "Usage: cmd [-j build_jobs] [-c configure] [-b build] [-t ctest] [-e benchmark] [-v verbose] [-u update_sources]"
    exit 1
    ;;
  esac
done

if [ "${IsVerbose}" == "yes" ]; then
  set -x
fi

TestBuildTool='make'
if command -v ninja >/dev/null; then
  CmakeGenerator="-GNinja"
  TestBuildTool='ninja'
fi

if ! command -v git-lfs >/dev/null; then
  echo "WARNING: git-lfs is not installed. Expect some tests to fail."
fi

if [ ! -d ${LLAMA_TESTS_LOG_LOCATION} ]; then
  mkdir -p ${LLAMA_TESTS_LOG_LOCATION}
fi

if [ ! -d ${LLAMA_SRC_DIR} ]; then
  echo "Cloning llama.cpp repository..."
  git clone https://github.com/ggml-org/llama.cpp.git src
elif [ "${DoUpdate}" == "yes" ]; then
  echo "Updating llama.cpp repository..."
  cd ${LLAMA_SRC_DIR}
  git pull
  cd ..
fi

echo "Configuring build with CMake..."
if [ "${DoConfigure}" == "yes" ]; then
  rm -rf ${LLAMA_BUILD_DIR}
  cmake -B build \
    -S src \
    -DCMAKE_PREFIX_PATH=${AOMP}/lib/cmake \
    -DGGML_HIP=On \
    -DCMAKE_BUILD_TYPE=${LLAMA_BUILD_MODE} \
    -DGPU_TARGETS=${LLAMA_GPU} \
    ${CmakeGenerator} \
    -DCMAKE_C_COMPILER=${AOMP}/bin/clang \
    -DCMAKE_CXX_COMPILER=${AOMP}/bin/clang++ \
    -DCMAKE_HIP_COMPILER=${AOMP}/bin/clang++
fi

if [ "${DoCompile}" == "yes" ]; then
  echo "Building LLaMA..."
  cmake --build ${LLAMA_BUILD_DIR} --parallel -j ${AOMP_BUILD_JOBS}
fi

if [ "${DoCTest}" == "yes" ]; then
  echo "Running tests..."
  cd ${LLAMA_BUILD_DIR}
  echo "Log in ${LLAMA_TESTS_LOG_LOCATION}/ctest.log"

  # Some model files are git-lfs and come from huggingface. They will auto-download during test
  ctest --output-on-failure 2>&1 | tee "${LLAMA_TESTS_LOG_LOCATION}/ctest.log"
fi

if [ "${DoBenchmark}" == "yes" ]; then
  echo "Running benchmark..."
  cd ${LLAMA_BUILD_DIR}
  # Download model from HF (this will make it avail in local cache); bench call requires local model file
  # llama-cli will turn on interactive mode, so echo /exit to it immediately
  echo "/exit" | ./bin/llama-cli -hf ${LLAMA_BENCH_HF_ID}

  # For the find command, we need to replace '/' with '_' in the LLAMA_BENCH_HF_ID
  ModelSearchPattern=${LLAMA_BENCH_HF_ID/\//\_}
  # Pick up first model with that name. Basically, pick up first quantization of that model
  LlamaModelPath=$(find "$LLAMA_CACHE" -maxdepth 1 -name "${ModelSearchPattern}*.gguf" 2>/dev/null | head -1)

  # Marker for external scripts
  echo "LLAMA_BENCHMARK_BEGIN" | tee "${LLAMA_TESTS_LOG_LOCATION}/llama-bench.log"
  # Run benchmark
  ./bin/llama-bench -ngl 999 -fa 1 -ub 2048 -m "$LlamaModelPath" 2>&1 | tee -a "${LLAMA_TESTS_LOG_LOCATION}/llama-bench.log"
fi

popd

#!/usr/bin/env bash

#
#Copyright © Advanced Micro Devices, Inc., or its affiliates.
#
#SPDX-License-Identifier:  MIT
#

# Build script for LLaMA with HIP support using AOMP compiler

ScriptDir=$(dirname "$(realpath "$0")")

# shellcheck source=/dev/null
. "${ScriptDir}"/aomp_common_vars

: "${ROCM_PATH:=$(realpath -m "${AOMP}/../..")}"
: "${AOMP_GPU:=gfx90a}"
: "${LLAMA_GPU:=$AOMP_GPU}"

: "${LLAMA_TLDIR:=$AOMP_REPOS_TEST/llama}"
: "${LLAMA_BUILD_DIR:=$LLAMA_TLDIR/build}"
: "${LLAMA_SRC_DIR:=$LLAMA_TLDIR/src}"
: "${LLAMA_BUILD_MODE:=Release}"
: "${LLAMA_TESTS_LOG_LOCATION:=$LLAMA_TLDIR/logs}"

# CTest 'test-backend-ops' exceeds the 1500s default timeout.
# Measured on MI350X: ~2500s
: "${LLAMA_CTEST_TIMEOUT:=3600}"

# Model to use in benchmarks (default is a smaller model)
: "${LLAMA_BENCH_HF_ID:=ggml-org/gemma-3-1b-it-GGUF}"
: "${LLAMA_CACHE:=$HOME/.cache/llama.cpp}"

# Add AOMP and ROCM_PATH to PATH and LD_LIBRARY_PATH and export them.
merge_variable_with_inputs PATH "${AOMP:+${AOMP}/bin}" "${ROCM_PATH:+${ROCM_PATH}/bin}"
merge_variable_with_inputs LD_LIBRARY_PATH \
  "${AOMP:+${AOMP}/lib}" \
  "${AOMP:+${AOMP}/lib/x86_64-unknown-linux-gnu}" \
  "${ROCM_PATH:+${ROCM_PATH}/lib}"

export PATH
export LD_LIBRARY_PATH
export ROCM_PATH

pushd "${AOMP_REPOS_TEST}" || exit
mkdir -p "${LLAMA_TLDIR}" && cd "${LLAMA_TLDIR}" || exit

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

if command -v ninja >/dev/null; then
  CmakeGenerator="-GNinja"
fi

if [ ! -d "${LLAMA_TESTS_LOG_LOCATION}" ]; then
  mkdir -p "${LLAMA_TESTS_LOG_LOCATION}"
fi

if [ ! -d "${LLAMA_SRC_DIR}" ]; then
  echo "Cloning llama.cpp repository..."
  git clone https://github.com/ggml-org/llama.cpp.git src
elif [ "${DoUpdate}" == "yes" ]; then
  echo "Updating llama.cpp repository..."
  cd "${LLAMA_SRC_DIR}" || exit
  git pull
  cd ..
fi

if ! command -v git-lfs >/dev/null; then
  echo "WARNING: git-lfs is not installed. Expect some tests to fail."
else
  # Ensure git-lfs is initialized and pulls any large files
  cd "${LLAMA_SRC_DIR}" || exit
  git lfs install
  git lfs pull
  cd ..
fi

if [ "${DoConfigure}" == "yes" ]; then
  echo "Configuring build with CMake..."
  rm -rf "${LLAMA_BUILD_DIR}"

  CMakeArgs=()
  if [ -n "${CmakeGenerator}" ]; then
    CMakeArgs+=("${CmakeGenerator}")
  fi

  # Determine the CMake module paths of required ROCm packages.
  CMakePrefixPath="${ROCM_PATH}"
  declare -A SeenPrefix=()
  for Package in hip hipblas rocblas; do
    if ! PackageCmakeDir=$(get_cmake_module_path "${Package}"); then
      Msg="ERROR: no CMake package '${Package}' below ${AOMP},"
      Msg+=" ${ROCM_PATH} or /opt/rocm"
      echo "${Msg}"
      exit 1
    fi

    # Anything found below the ROCm under test needs no extra prefix.
    # Removing a prefix that is present shortens the path, so a path that
    # comes back unchanged did not start with it.
    if [ "${PackageCmakeDir#"${AOMP}"/}" != "${PackageCmakeDir}" ] ||
       [ "${PackageCmakeDir#"${ROCM_PATH}"/}" != "${PackageCmakeDir}" ]; then
      continue
    fi

    Msg="WARNING: ${ROCM_PATH} does not provide ${Package},"
    Msg+=" using ${PackageCmakeDir}"
    echo "${Msg}"

    # Two packages commonly resolve to the same place; append it only once.
    if [ -z "${SeenPrefix[${PackageCmakeDir}]:-}" ]; then
      SeenPrefix["${PackageCmakeDir}"]=1
      CMakePrefixPath+=";${PackageCmakeDir}"
    fi
  done
  unset SeenPrefix

  CMakeArgs+=("-S" "src")
  CMakeArgs+=("-B" "build")
  CMakeArgs+=("-DCMAKE_PREFIX_PATH=${CMakePrefixPath}")
  CMakeArgs+=("-DGGML_HIP=On")
  CMakeArgs+=("-DCMAKE_BUILD_TYPE=${LLAMA_BUILD_MODE}")
  CMakeArgs+=("-DGPU_TARGETS=${LLAMA_GPU}")
  CMakeArgs+=("-DCMAKE_C_COMPILER=${AOMP}/bin/clang")
  CMakeArgs+=("-DCMAKE_CXX_COMPILER=${AOMP}/bin/clang++")
  CMakeArgs+=("-DCMAKE_HIP_COMPILER=${AOMP}/bin/clang++")

  # CMake modules export their whole include directory, HIP headers included,
  # which would shadow the ROCm under test and e.g. its HIP headers.
  add_cmake_rocm_header_priority_args CMakeArgs "${ROCM_PATH}"

  printf 'cmake'; printf ' %q' "${CMakeArgs[@]}"; printf '\n'
  cmake "${CMakeArgs[@]}" 2>&1 |
    tee "${LLAMA_TESTS_LOG_LOCATION}/cmake-configure.log"

  # Make sure the ROCm header priority is preserved.
  check_cmake_rocm_header_priority "${LLAMA_BUILD_DIR}" "${ROCM_PATH}" || exit 1
fi

if [ "${DoCompile}" == "yes" ]; then
  echo "Building LLaMA..."
  cmake --build "${LLAMA_BUILD_DIR}" --parallel -j "${AOMP_BUILD_JOBS}"
fi

if [ "${DoCTest}" == "yes" ]; then
  echo "Running tests..."
  cd "${LLAMA_BUILD_DIR}" || exit
  echo "Log in ${LLAMA_TESTS_LOG_LOCATION}/ctest.log"

  # Some model files are git-lfs and come from huggingface. They will auto-download during test
  ctest --output-on-failure --timeout "${LLAMA_CTEST_TIMEOUT}" 2>&1 |
    tee "${LLAMA_TESTS_LOG_LOCATION}/ctest.log"
fi

run_llama_bench() {
  ./bin/llama-bench "$@" 2>&1 | tee -a "${LLAMA_TESTS_LOG_LOCATION}/llama-bench.log"
  BenchStatus=${PIPESTATUS[0]}

  if [ "${BenchStatus}" -ne 0 ]; then
    echo "ERROR: llama-bench failed with exit code ${BenchStatus}" | tee -a "${LLAMA_TESTS_LOG_LOCATION}/llama-bench.log"
    return "${BenchStatus}"
  fi
}

if [ "${DoBenchmark}" == "yes" ]; then
  echo "Running benchmark..."
  cd "${LLAMA_BUILD_DIR}" || exit

  # Get cache directory from llama-cli if supported by the local build.
  CacheListOutput=$(./bin/llama-cli --cache-list 2>&1 || true)
  CacheDir=$(echo "${CacheListOutput}" | grep "model cache directory:" | sed 's/.*: //')
  : "${CacheDir:=${LLAMA_CACHE}}"

  # Find requested model by converting HF ID to filename pattern (user/model -> user_model)
  SearchPattern="${LLAMA_BENCH_HF_ID//\//_}"
  LlamaModelPath=$(find "${CacheDir}" \( -type f -o -xtype f \) -name "${SearchPattern}*.gguf" 2>/dev/null | head -1)

  # Fallback: use all available .gguf files in cache
  if [ -z "${LlamaModelPath}" ]; then
    echo "Requested model not found, using all cached models"
    mapfile -t ModelPaths < <(find "${CacheDir}" \( -type f -o -xtype f \) -name "*.gguf" 2>/dev/null)
  else
    ModelPaths=("${LlamaModelPath}")
  fi

  # Marker for external scripts
  echo "LLAMA_BENCHMARK_BEGIN" | tee "${LLAMA_TESTS_LOG_LOCATION}/llama-bench.log"

  if [ ${#ModelPaths[@]} -eq 0 ] && ./bin/llama-bench --help 2>&1 | grep -q -- "--hf-repo"; then
    # Let llama-bench resolve/download the HF model directly.  Using llama-cli as
    # a prefetch step can hang in ROCm/KFD waits on cold cache.
    run_llama_bench -hf "${LLAMA_BENCH_HF_ID}" -ngl 999 -fa 1 -ub 2048 || exit $?
  else
    if [ ${#ModelPaths[@]} -eq 0 ]; then
      echo "ERROR: No model files found in cache directory: ${CacheDir}"
      exit 1
    fi

    # Run benchmark for each model
    for LlamaModelPath in "${ModelPaths[@]}"; do
      echo "Benchmarking: ${LlamaModelPath}" | tee -a "${LLAMA_TESTS_LOG_LOCATION}/llama-bench.log"
      run_llama_bench -ngl 999 -fa 1 -ub 2048 -m "${LlamaModelPath}" || exit $?
    done
  fi
fi

popd || exit

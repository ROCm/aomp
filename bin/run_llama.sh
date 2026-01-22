#!/usr/bin/env bash

# Build script for LLaMA with HIP support using AOMP compiler

. aomp_common_vars

: AOMP_GPU=${AOMP_GPU:=gfx90a}
: ${LLAMA_GPU:=$AOMP_GPU}

: ${LLAMA_TLDIR:=$AOMP_REPOS_TEST/llama}
: ${LLAMA_BUILD_DIR:=$LLAMA_TLDIR/build}
: ${LLAMA_SRC_DIR:=$LLAMA_TLDIR/src}
: ${LLAMA_BUILD_MODE:=Release}

pushd ${AOMP_REPOS_TEST}
mkdir -p ${LLAMA_TLDIR} && cd ${LLAMA_TLDIR}

# Run CMake configuration
DoConfigure='no'

# Run build command
DoCompile='no'

# Run ctest
DoCTest='no'

IsVerbose='no'

while getopts "j:cbtv" opt; do
  case ${opt} in
    j ) AOMP_BUILD_JOBS=${OPTARG} ;;
    c ) DoConfigure='yes' ;;
    b ) DoCompile='yes' ;;
    t ) DoCTest='yes' ;;
    v ) IsVerbose='yes' ;;
    \? ) echo "Usage: cmd [-j build_jobs] [-c configure] [-b build] [-t ctest]"
         exit 1 ;;
  esac
done

if [ "${IsVerbose}" == "yes" ]; then
  set -x
fi

if [ ! -d ${LLAMA_SRC_DIR} ]; then
  echo "Cloning llama.cpp repository..."
  git clone https://github.com/ggml-org/llama.cpp.git src
else
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
    -GNinja \
    -DCMAKE_C_COMPILER=${AOMP}/bin/clang \
    -DCMAKE_CXX_COMPILER=${AOMP}/bin/clang++ \
    -DCMAKE_HIP_COMPILER=${AOMP}/bin/clang++
fi

if [ "${DoCompile}" == "yes" ]; then
  echo "Building LLaMA..."
  cmake --build ${LLAMA_BUILD_DIR} --parallel
fi

if [ "${DoCTest}" == "yes" ]; then
  echo "Running tests..."
  cd ${LLAMA_BUILD_DIR}
  ctest --output-on-failure
fi

popd

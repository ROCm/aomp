#!/bin/bash
#
# build_llvm-flang-rt-host-dev.sh
#
#   Standalone script to build the flang runtime with host-device support
#   This script should only be run after build_aomp.sh
#   Installs in: ${AOMP}/lib/libflang_rt.hostdevice.a
#
# References:
#
# https://libc.llvm.org/gpu/building.html
# https://flang.llvm.org/docs/GettingStarted.html#openmp-target-offload-build
# https://github.com/llvm/llvm-project/blob/main/flang/docs/GettingStarted.md
#
# --- Start standard header to set AOMP environment variables ----
realpath=$(realpath "$0")
thisdir=$(dirname "$realpath")
. "$thisdir/aomp_common_vars"
# --- end standard header ----

# for TheRock to override with amd-llvm
export AOMP_NAME_LLVM_PROJECT=${AOMP_NAME_LLVM_PROJECT:-llvm-project}
export BUILD_AOMP_SUBDIR=${BUILD_AOMP_SUBDIR:-build/llvm-project}

echo "-----------------------------------------------------------------------------"
echo "Building flang-runtime for device"
echo "AOMP               = $AOMP"
echo "AOMP_REPOS         = $AOMP_REPOS"
echo "BUILD_AOMP         = $BUILD_AOMP"

CMAKE_C_COMPILER="$AOMP/bin/clang"
CMAKE_CXX_COMPILER="$AOMP/bin/clang++"

if [ -z ${AOMP+x} ]; then
    echo "Error: AOMP must be defined"
    exit 0
fi
if [ -z ${AOMP_REPOS+x} ]; then
    echo "Error: AOMP_REPOS must be defined"
    exit 0
fi
if [ ! -x "$CMAKE_C_COMPILER" ]; then
    # try again by adding the llvm subdirectory
    AOMP="$AOMP/llvm"
    CMAKE_C_COMPILER="$AOMP/bin/clang"
    CMAKE_CXX_COMPILER="$AOMP/bin/clang++"

    if [ ! -x "$CMAKE_C_COMPILER" ]; then
        echo "Error: $CMAKE_C_COMPILER not found"
        exit 0
    fi
fi
if [ ! -x "$CMAKE_CXX_COMPILER" ]; then
    echo "Error: $CMAKE_CXX_COMPILER not found"
    exit 0
fi

BUILD_DIR=$BUILD_AOMP/$BUILD_AOMP_SUBDIR
BUILD_DIR_FRT=$BUILD_AOMP/build/flang-runtime/flang-rt/lib
OMPRUNTIME_DIR=$BUILD_DIR/runtimes/runtimes-bins/openmp/runtime/src
INSTALL_DIR=${INSTALL_DIR:-$AOMP}
SUFFIX=${SUFFIX:-}

# generate ARCH_LIST from GFXLIST
ARCH_LIST=$(echo "$GFXLIST" | tr ' ' ',')
#BUILD_TYPE=${BUILD_TYPE:-Release}    # note: hits backend assert
BUILD_TYPE=${BUILD_TYPE:-}

echo "BUILD_DIR          = $BUILD_DIR"
echo "BUILD_DIR_FRT      = $BUILD_DIR_FRT"
echo "OMPRUNTIME_DIR     = $OMPRUNTIME_DIR"
echo "INSTALL_DIR        = $INSTALL_DIR"
echo "CMAKE_C_COMPILER   = $CMAKE_C_COMPILER"
echo "CMAKE_CXX_COMPILER = $CMAKE_CXX_COMPILER"
echo "GFXLIST            = $GFXLIST"
echo "BUILD_TYPE         = $BUILD_TYPE"
echo "SUFFIX             = $SUFFIX"

echo "Sleeping 5 sec..."
sleep 5

mkdir -p "$BUILD_AOMP"
cd "$BUILD_AOMP" || exit
mkdir -p build
cd build || exit
rm -rf flang-runtime
mkdir flang-runtime
cd flang-runtime || exit

if [ "$AOMP_USE_NINJA" == 0 ] ; then
    AOMP_SET_NINJA_GEN=()
else
    AOMP_SET_NINJA_GEN=(-G Ninja)
fi

# Notes:
#   -DFLANG_RT_INCLUDE_TESTS=OFF     # avoids needing CUDA toolchain
#
if [ ${BUILD_TYPE+x} ]; then
    CM_BUILD_TYPE="-DCMAKE_BUILD_TYPE='$BUILD_TYPE'"
fi
${AOMP_CMAKE} "${AOMP_SET_NINJA_GEN[@]}" $CM_BUILD_TYPE \
    -DLLVM_ENABLE_RUNTIMES=flang-rt \
    -DFLANG_RT_EXPERIMENTAL_OFFLOAD_SUPPORT="OpenMP" \
    -DFLANG_RT_INCLUDE_TESTS=OFF \
    -DCMAKE_C_COMPILER="$CMAKE_C_COMPILER" \
    -DCMAKE_CXX_COMPILER="$CMAKE_CXX_COMPILER" \
    -DFLANG_RT_DEVICE_ARCHITECTURES="$ARCH_LIST" \
    -DCMAKE_EXPORT_COMPILE_COMMANDS=ON \
    -DFLANG_RT_EMBED_GPU_LLVM_IR=OFF \
    "$AOMP_REPOS/$AOMP_NAME_LLVM_PROJECT/runtimes"

$AOMP_NINJA_BIN --version
$AOMP_NINJA_BIN -j "$AOMP_JOB_THREADS" flang-rt
mystat=$?
allstat=$((allstat+mystat))
echo "status: $mystat"

cmd="cp $BUILD_DIR_FRT/libflang_rt.runtime.a $INSTALL_DIR/lib/libflang_rt.hostdevice${SUFFIX}.a"
echo "$cmd"
$cmd
mystat=$?
allstat=$((allstat+mystat))
echo "status: $mystat"

echo "allstat: $allstat"
# Note: Currently ignore build status of fortran-rt-host-dev
# If this fails to build, don't trigger nightly compiler staging to fail
exit 0

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
realpath=`realpath $0`
thisdir=`dirname $realpath`
. $thisdir/aomp_common_vars
# --- end standard header ----

echo "-----------------------------------------------------------------------------"
echo "Building flang-runtime for device"
cat <<EOD

Note: If working on amd-staging, the following patch is currently needed:
      cd $AOMP_REPOS/llvm-project
      patch -p1 < $AOMP_REPOS/aomp/bin/patches/llvm-flang-rt-host-dev.patch

EOD
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
if [ ! -x $CMAKE_C_COMPILER ]; then
    # try again by adding the llvm subdirectory
    AOMP="$AOMP/llvm"
    CMAKE_C_COMPILER="$AOMP/bin/clang"
    CMAKE_CXX_COMPILER="$AOMP/bin/clang++"

    if [ ! -x $CMAKE_C_COMPILER ]; then
        echo "Error: $CMAKE_C_COMPILER not found"
        exit 0
    fi
fi
if [ ! -x $CMAKE_CXX_COMPILER ]; then
    echo "Error: $CMAKE_CXX_COMPILER not found"
    exit 0
fi

BUILD_DIR=$BUILD_AOMP/build/llvm-project
BUILD_DIR_FRT=$BUILD_AOMP/build/flang-runtime/
OMPRUNTIME_DIR=$BUILD_DIR/runtimes/runtimes-bins/openmp/runtime/src
INSTALL_DIR=$AOMP

# generate ARCH_LIST from GFXLIST
ARCH_LIST=`echo $GFXLIST | tr ' ' ','`

echo "BUILD_DIR          = $BUILD_DIR"
echo "BUILD_DIR_FRT      = $BUILD_DIR_FRT"
echo "OMPRUNTIME_DIR     = $OMPRUNTIME_DIR"
echo "INSTALL_DIR        = $INSTALL_DIR"
echo "CMAKE_C_COMPILER   = $CMAKE_C_COMPILER"
echo "CMAKE_CXX_COMPILER = $CMAKE_CXX_COMPILER"
echo "GFXLIST            = $GFXLIST"

echo "Sleeping 5 sec..."
sleep 5

cd $AOMP_REPOS
mkdir -p build
cd build
rm -rf flang-runtime
mkdir flang-runtime
cd flang-runtime

if [ "$AOMP_USE_NINJA" == 0 ] ; then
    AOMP_SET_NINJA_GEN=""
else
    AOMP_SET_NINJA_GEN="-G Ninja"
fi
${AOMP_CMAKE} $AOMP_SET_NINJA_GEN \
    -DFLANG_EXPERIMENTAL_OMP_OFFLOAD_BUILD="host_device" \
    -DCMAKE_C_COMPILER=$CMAKE_C_COMPILER \
    -DCMAKE_CXX_COMPILER=$CMAKE_CXX_COMPILER \
    -DCMAKE_EXPORT_COMPILE_COMMANDS=ON \
    "-DCMAKE_C_FLAGS=-I$OMPRUNTIME_DIR" \
    "-DCMAKE_CXX_FLAGS=-I$OMPRUNTIME_DIR" \
    -DFLANG_OMP_DEVICE_ARCHITECTURES="$ARCH_LIST" \
    $AOMP_REPOS/llvm-project/flang/runtime

$AOMP_NINJA_BIN --version
$AOMP_NINJA_BIN -j $AOMP_JOB_THREADS FortranRuntime
mystat=$?
allstat=$(($allstat+$mystat))
echo "status: $mystat"

if [ -e "$BUILD_DIR_FRT/libflang_rt.runtime.a" ]; then
    cmd="cp $BUILD_DIR_FRT/libflang_rt.runtime.a $INSTALL_DIR/lib/libflang_rt.hostdevice.a"
    echo $cmd
    $cmd
    mystat=$?
    allstat=$(($allstat+$mystat))
    echo "status: $mystat"
fi

echo "allstat: $allstat"
# Note: Currently ignore build status of fortran-rt-host-dev
# If this fails to build, don't trigger nightly compiler staging to fail
exit 0

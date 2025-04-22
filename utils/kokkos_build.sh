#!/bin/bash
# Copyright(C) 2020 Advanced Micro Devices, Inc. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies
# of the Software, and to permit persons to whom the Software is furnished to do so,
# subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
# IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM,
# DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE,
# ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
# IN THE SOFTWARE.
#
#  kokkos_build.sh: Script to clone and build kokkos for a specific GPU
#                   This will build kokkos in directory $HOME/kokkos_build.<GPUNAME>
#
#
#  Written by Greg Rodgers  Gregory.Rodgers@amd.com, Jan-Patrick Lehr JanPatrick.Lehr@amd.com
#
PROGVERSION="X.Y-Z"
#

function print_info() {
  toPrint="$1"

  echo "[Info]: $toPrint"
}
function print_error() {
  toPrint="$1"

  echo "[Error]: $toPrint"
}
echo "$AOMP"
AOMP="${AOMP:-AOMP_INSTALL_DIR}"
echo "$AOMP"
if [ ! -d $AOMP ] ; then
   print_error "AOMP is not installed in ${AOMP}. Please set the environment variable."
   exit 1
fi

KOKKOS_SOURCE_PREFIX=${KOKKOS_SOURCE_PREFIX:-$HOME/git}
KOKKOS_SOURCE_DIR=${KOKKOS_SOURCE_DIR:-$KOKKOS_SOURCE_PREFIX/kokkos}
KOKKOS_URL=https://github.com/kokkos/kokkos.git
# Per default (for now) we want to use release 3.7.00 to establish a working baseline.
KOKKOS_TAG=${_KOKKOS_TAG_:-3.7.01}
# For development we also want to pull-in (most recent) devel
KOKKOS_BRANCH=${_KOKKOS_BRANCH_:-develop}
KOKKOS_SHA=7c76889 # This is pretty old
KOKKOS_SHA=${_KOKKOS_SHA_:-}
NUM_THREADS=${NUM_THREADS:-8}

COMPILERNAME_TO_USE=${_COMPILER_TO_USE_:-clang++}


if [[ "$AOMP" =~ "opt" ]]; then
  # xargs trims the string off whitespaces
  export DETECTED_GPU=$($AOMP/../bin/rocminfo | grep -m 1 -E gfx[^0]{1}.{2} | awk '{print $2}')
  print_info "Set AOMP_GPU with rocminfo: $DETECTED_GPU"
  #print_info "Set AOMP_GPU with rocm_agent_enumerator.
  #export DETECTED_GPU=$($AOMP/../bin/rocm_agent_enumerator | grep -m 1 -E gfx[^0]{1}.{2})
else
  print_info "Set AOMP_GPU with offload-arch."
  if [ -a $AOMP/bin/rocminfo ]; then
    export DETECTED_GPU=$($AOMP/bin/rocminfo | grep -m 1 -E gfx[^0]{1}.{2} | awk '{print $2}')
  else
    export DETECTED_GPU=$($AOMP/bin/offload-arch | grep -m 1 -E gfx[^0]{1}.{2})
  fi
fi

AOMP_VERSION=$($AOMP/bin/${COMPILERNAME_TO_USE} --version | head -n 1)

AOMP_GPU=${AOMP_GPU:-$DETECTED_GPU}

# Determine if AOMP_GPU is supported in KOKKOS. Currently looks for AMD only.
kokkos_regex="gfx(.*)"
supported_arch_vega="900 906 908 90a"
supported_arch_navi="1030 1100"
declare -A arch_array
print_info "Supported KOKKOS GFX: $supported_arch_vega $supported_arch_navi"

# Store arch in associative array
for arch in $supported_arch_vega; do
  # Add name used in Kokkos CMake for Architecture
  arch_str="VEGA${arch^^}"
  arch_array[$arch]=$arch_str
done

for arch in $supported_arch_navi; do
  # Add name used in Kokkos CMake for Architecture
  arch_str="NAVI${arch^^}"
  arch_array[$arch]=$arch_str
done

if [[ "$AOMP_GPU" =~ $kokkos_regex ]]; then
  matched_arch=${BASH_REMATCH[1]}
else
  print_error "Error: Cannot determine KOKKOS_ARCH"
fi

# Check if matched_arch is present in array
if [[  -v arch_array[$matched_arch] ]]; then
  # Get Kokkos name for arch
  KOKKOS_ARCH=${KOKKOS_ARCH:-${arch_array["$matched_arch"]}}
else
  print_error "Error: gfx${matched_arch} is currently not supported in KOKKOS"
  exit 1
fi

print_info "Using configuration"
print_info "AOMP_GPU    = $AOMP_GPU"
print_info "KOKKOS_TAG  = $KOKKOS_TAG"
print_info "KOKKOS_ARCH = $KOKKOS_ARCH"
print_info "KOKKOS_SOURCE = $KOKKOS_SOURCE_DIR"
print_info "AOMP        = $AOMP"
print_info "AOMP Version = $AOMP_VERSION"

if [ "$AOMP_GPU" == "" ] || [ "$AOMP_GPU" == "unknown" ]; then
  print_error "Error: AOMP_GPU not properly set...exiting."
  exit 1
fi

KOKKOS_BUILD_PREFIX=${KOKKOS_BUILD_PREFIX:-$HOME}

if [ -f /usr/local/cmake/bin/cmake ] ; then
  AOMP_CMAKE=${AOMP_CMAKE:-/usr/local/cmake/bin/cmake}
else
  AOMP_CMAKE=${AOMP_CMAKE:-cmake}
fi

if [ "$1" == "hip" ] ; then
   kokkos_backend="hip"
   KOKKOS_BUILD_DIR=${KOKKOS_BUILD_DIR:-$KOKKOS_BUILD_PREFIX/kokkos-${KOKKOS_TAG}_build_hip.$AOMP_GPU}
   KOKKOS_INSTALL_DIR=${KOKKOS_INSTALL_DIR:-$KOKKOS_BUILD_PREFIX/kokkos-${KOKKOS_TAG}_hip.$AOMP_GPU}
else
   kokkos_backend="openmp"
   KOKKOS_BUILD_DIR=${KOKKOS_BUILD_DIR:-$KOKKOS_BUILD_PREFIX/kokkos-${KOKKOS_TAG}_build_omp.$AOMP_GPU}
   KOKKOS_INSTALL_DIR=${KOKKOS_INSTALL_DIR:-$KOKKOS_BUILD_PREFIX/kokkos-${KOKKOS_TAG}_omp.$AOMP_GPU}
fi

# Clean install, build and source directories
if [ "$1" == "clean" ] ; then
  print_info "Cleaning:"
  print_info "SOURCE_DIR: $KOKKOS_SOURCE_DIR"
  print_info "BUILD_DIR: $KOKKOS_BUILD_DIR"
  print_info "INSTALL_DIR: $KOKKOS_INSTALL_DIR"
  rm -rf $KOKKOS_SOURCE_DIR
  rm -rf $KOKKOS_BUILD_DIR
  rm -rf $KOKKOS_INSTALL_DIR
  exit 0
fi

# This patch is required for builds of Clang/LLVM w/ assertions enabled for a Clang assertion failing
# otherwise: Non-trivially copyable data types used by Kokkos.
function patchkokkos(){
patchfile1=/tmp/kokkos1_$$.patch
/bin/cat >$patchfile1 <<"EOF"
From e1a19dda160751eefb6b70a204a21db7a93c48b1 Mon Sep 17 00:00:00 2001
From: JP Lehr <JanPatrick.Lehr@amd.com>
Date: Mon, 31 Oct 2022 15:17:58 +0000
Subject: [PATCH] Only to create a path

---
core/unit_test/TestNonTrivialScalarTypes.hpp | 2 ++
1 file changed, 2 insertions(+)

diff --git a/core/unit_test/TestNonTrivialScalarTypes.hpp b/core/unit_test/TestNonTrivialScalarTypes.hpp
index 02064d2fc..e593e7e3d 100644
--- a/core/unit_test/TestNonTrivialScalarTypes.hpp
+++ b/core/unit_test/TestNonTrivialScalarTypes.hpp
@@ -158,10 +158,12 @@ struct array_reduce {
   array_reduce() {
     for (int i = 0; i < N; i++) data[i] = scalar_t();
   }
+#if 0
   KOKKOS_INLINE_FUNCTION
   array_reduce(const array_reduce &rhs) {
     for (int i = 0; i < N; i++) data[i] = rhs.data[i];
   }
+#endif
   KOKKOS_INLINE_FUNCTION
   array_reduce(const scalar_t value) {
     for (int i = 0; i < N; i++) data[i] = scalar_t(value);
---
2.25.1

EOF
  print_info "patch -p1 < $patchfile1"
  patch -p1 < $patchfile1
  if [[ $? -ne 0 ]]; then
    print_error "Patching unsuccessful"
    exit -1
  fi
 rm $patchfile1
}

mkdir -p $KOKKOS_SOURCE_PREFIX
cd $KOKKOS_SOURCE_PREFIX || exit 1
if [ ! -d $KOKKOS_SOURCE_DIR ] ; then
   print_info "git clone $KOKKOS_URL $KOKKOS_SOURCE_DIR"
   git clone $KOKKOS_URL $KOKKOS_SOURCE_DIR
   if [ $? != 0 ] ; then
      echo
      print_error "ERROR: Could not git clone $KOKKOS_URL "
      exit 1
   fi
   cd $KOKKOS_SOURCE_DIR || exit 1
   print_info "Checking out Kokkos branch: ${KOKKOS_BRANCH}"
   git checkout $KOKKOS_BRANCH
   if [ $? != 0 ] ; then
      echo
      print_error "ERROR: Could not check out $KOKKOS_BRANCH"
      exit 1
   fi
   if [[ ! -z $KOKKOS_TAG ]]; then
     print_info "Switch to Kokkos tag: ${KOKKOS_TAG}"
     git checkout $KOKKOS_TAG
   elif [[ ! -z $KOKKOS_SHA ]]; then
     print_info "Switch to Kokkos SHA: ${KOKKOS_SHA}"
     git checkout $KOKKOS_SHA
   else
     print_info "Running on $KOKKOS_BRANCH HEAD"
   fi
   if [ "$PATCH" != 0 ]; then
     print_info "patching..."
     patchkokkos
   else
     print_info "PATCH=0 - not patching KOKKOS"
   fi
else
   echo
   echo "  WARNING: Directory $KOKKOS_SOURCE_DIR already exists."
   echo "           No changes will be made to the sources"
   echo
fi

if [ -d "$KOKKOS_BUILD_DIR" ] ; then
   echo
   echo " FRESH START:  Removing $KOKKOS_BUILD_DIR"
   echo "               rm -rf $KOKKOS_BUILD_DIR"
   echo
   rm -rf  $KOKKOS_BUILD_DIR
fi
if [ -d "$KOKKOS_INSTALL_DIR" ] ; then
   echo "               rm -rf $KOKKOS_INSTALL_DIR"
   echo
   rm -rf  $KOKKOS_INSTALL_DIR
fi

print_info "$KOKKOS_BUILD_DIR"

mkdir -p $KOKKOS_BUILD_DIR
cd "$KOKKOS_BUILD_DIR"

UNAMEP=`uname -m`
AOMP_CPUTARGET="${UNAMEP}-pc-linux-gnu"
if [ "$UNAMEP" == "ppc64le" ] ; then
   AOMP_CPUTARGET="ppc64le-linux-gnu"
fi

export PATH=$HOME/local/install/cmake/bin:$PATH
which cmake

if [ "$kokkos_backend" == "hip" ] ; then
  export PATH=$AOMP/bin:$PATH
  export ROCM_PATH=$AOMP # Kokkos searches this for HIP parts
  ARGS=(
    -D CMAKE_BUILD_TYPE=Debug
    -D CMAKE_INSTALL_PREFIX=$KOKKOS_INSTALL_DIR
    -D CMAKE_CXX_STANDARD=17
    -D CMAKE_CXX_EXTENSIONS=OFF
    -D CMAKE_CXX_COMPILER=$AOMP/bin/clang++
    -D Kokkos_ARCH_NATIVE=ON
    -D Kokkos_ARCH_"$KOKKOS_ARCH"=ON
    -D Kokkos_ENABLE_OPENMP=ON
    -D Kokkos_ENABLE_HIP=ON
    -D Kokkos_ENABLE_COMPILER_WARNINGS=ON
    -D Kokkos_ENABLE_TESTS=OFF
  )

else
  #AOMP=$HOME/llvm-AMDGPU-NG
   # ensure kokkos finds AOMP clang first
   export PATH=$AOMP/bin:$PATH
   ARGS=(
    -D CMAKE_BUILD_TYPE=Release
    -D CMAKE_CXX_STANDARD=17
    -D CMAKE_CXX_EXTENSIONS=OFF
    -D CMAKE_INSTALL_PREFIX=$KOKKOS_INSTALL_DIR
    -D CMAKE_CXX_COMPILER=$AOMP/bin/${COMPILERNAME_TO_USE}
    #-D CMAKE_CXX_FLAGS="-mllvm -time-passes -mllvm -openmp-opt-max-iterations=10"
    -D CMAKE_CXX_FLAGS="-mllvm -time-passes"
    -D CMAKE_VERBOSE_MAKEFILE=ON
    -D Kokkos_ARCH_NATIVE=ON
    -D Kokkos_ARCH_"$KOKKOS_ARCH"=ON
    -D Kokkos_ENABLE_OPENMP=ON
    -D Kokkos_ENABLE_OPENMPTARGET=ON
    -D Kokkos_ENABLE_COMPILER_WARNINGS=ON
    -D Kokkos_ENABLE_TESTS=ON
   )

fi

   print_info "Running cmake via"
   echo "cmake ${ARGS[@]} $KOKKOS_SOURCE_DIR"
   cmake "${ARGS[@]}" $KOKKOS_SOURCE_DIR

if [ $? != 0 ] ; then
   echo
   echo "ERROR in Kokkos cmake"
   echo "If HWLOC is not found, set environment variable HWLOC_DIR=/path/to/hwloc."
   echo
   echo "cmake ${ARGS[@]} $KOKKOS_SOURCE_DIR"
   exit 1
fi

echo
echo "CMAKE done in directory $KOKKOS_BUILD_DIR"
echo
print_info "Starting build ..."

print_info make -j$NUM_THREADS
make --output-sync=recurse -j$NUM_THREADS 2>&1 | tee kokkos-build.log

if [ $? != 0 ] ; then
   print_error "ERROR in Kokkos build"
   exit 1
fi

make install
if [ $? != 0 ] ; then
   print_error "ERROR in Kokkos install"
   exit 1
fi
echo
echo " Kokkos Version $KOKKOS_SHA installed successfully into directory "
echo " $KOKKOS_INSTALL_DIR"
echo

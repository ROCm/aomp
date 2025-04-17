#!/bin/bash
# Copyright(C) 2023 Advanced Micro Devices, Inc. All rights reserved.
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
#  Written by Jan-Patrick Lehr <JanPatrick.Lehr@amd.com>
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

AOMP="${AOMP:-_AOMP_INSTALL_DIR_}"

if [ ! -d $AOMP ] ; then
   print_error "AOMP is not installed in ${AOMP}. Please set the environment variable."
   exit 1
fi

GIT_DIR=${GIT_DIR:-$HOME/git}
KOKKOS_SOURCE_DIR=${KOKKOS_SOURCE_DIR:-$GIT_DIR/kokkos}
KOKKOS_EXAMPLES_SOURCE_DIR=${KOKKOS_EXAMPLES_SOURCE_DIR:-$GIT_DIR/kokkos-openmptarget-examples}
KOKKOS_EXAMPLES_REPO=https://github.com/kokkos/kokkos-openmptarget-examples.git
NUM_THREADS=${NUM_THREADS:-8}

COMPILERNAME_TO_USE=${_COMPILER_TO_USE_:-clang++}
AOMP_VERSION=$($AOMP/bin/${COMPILERNAME_TO_USE} --version | head -n 1)

cd $GIT_DIR || exit 1

if [ "$1" == "clean" ]; then
  print_info "Removing $KOKKOS_EXAMPLES_SOURCE_DIR"
  rm -rf $KOKKOS_EXAMPLES_SOURCE_DIR
  exit 0
fi

# Get the source code
git clone $KOKKOS_EXAMPLES_REPO $KOKKOS_EXAMPLES_SOURCE_DIR

# Change to the directory
cd $KOKKOS_EXAMPLES_SOURCE_DIR || exit 1

cd cgsolve || exit 1

# The Makefile in the repo does not make use of a Kokkos installation
# Instead, it needs to be pointed to the Kokkos repository and fed additional config flags
# For now, we simply stick with what the Kokkos developers did, but we may want to change it
# But one thing we may need to change is the compiler name. let's do brute force for now.
sed -i "s/amdclang++/${COMPILERNAME_TO_USE}/g" ../Makefile.inc
# Do not use debug info for the time being.
sed -i "s/-O3 -g/-O3/g" Makefile
# Replace hard coded gfx90a with AOMP_GPU env var
echo "updating Makefile.inc for AOMP_GPU=$AOMP_GPU"
sed -i "s/-march=gfx90a/-march=$AOMP_GPU/" ../Makefile.inc

cmd="PATH=$AOMP/bin:$PATH CXX=clang++ make KOKKOS_PATH=$KOKKOS_SOURCE_DIR arch=MI250x backend=ompt comp=rocmclang"

print_info "$cmd"
PATH=$AOMP/bin:$PATH CXX=clang++ make KOKKOS_PATH=$KOKKOS_SOURCE_DIR arch=MI250x backend=ompt comp=rocmclang -j${NUM_THREADS}
#OFFLOAD_FLAGS='-ffast-math -fopenmp-target-fast'



#!/bin/bash
#
#  File: build_emissary.sh with symlink to build_emissary_mpi.sh
#
# MIT License
#
# Copyright (c) 2017 Advanced Micro Devices, Inc. All Rights Reserved.
#
# Permission is hereby granted, free of charge, to any person
# obtaining a copy of this software and associated documentation
# files (the "Software"), to deal in the Software without
# restriction, including without limitation the rights to use, copy,
# modify, merge, publish, distribute, sublicense, and/or sell copies
# of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be
# included in all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
# EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
# MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
# NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS
# BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN
# ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
# CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

# --- Start standard header to set AOMP environment variables ----
realpath=$(realpath "$0")
thisdir=$(dirname "$realpath")
. "$thisdir/aomp_common_vars"
# --- end standard header ----

_sname=${0##*/}
BUILD_DIR=${BUILD_AOMP}
declare -a extra_cmake_opts=()

if [ "$_sname" == "build_emissary_mpi.sh" ] ; then 
  EMISSARY_REPO_DIR=$AOMP_REPOS/emissary/MPI
  extra_cmake_opts+=("-DLLVM_EXTERNAL_EMISSARY_MPI_INSTALL=$HOME/local/rocmopenmpi")
  BUILD_EMISSARY_DIR="$BUILD_DIR/build/emissary_mpi"
elif [ "$_sname" == "build_emissary_hdf5.sh" ] ; then 
  EMISSARY_REPO_DIR=$AOMP_REPOS/emissary/HDF5
  BUILD_EMISSARY_DIR="$BUILD_DIR/build/emissary_hdf5"
else
  echo "ERROR: You must run build_emissary_mpi.sh"
  echo "        or build_emissary_hdf5.sh"
  exit
fi
echo "BUILD_EMISSARY_DIR=$BUILD_EMISSARY_DIR"

echo "INFO: Getting latest sources for emissary in dir \"$EMISSARY_REPO_DIR\""
if [ -d "$EMISSARY_REPO_DIR" ] ; then
  echo cd "$EMISSARY_REPO_DIR"
  cd "$EMISSARY_REPO_DIR" || exit
  echo git pull
  git pull 
else 
  echo cd "$AOMP_REPOS"
  cd "$AOMP_REPOS" || exit
  echo git clone git@github.com:rocm/emissary
  git clone git@github.com:rocm/emissary
  cd "$EMISSARY_REPO_DIR" || exit
fi


BUILDTYPE="Release"

# Install EMISSARY in the compiler directory of ROCm
INSTALL_EMISSARY=${INSTALL_EMISSARY:-$AOMP_INSTALL_DIR}/lib/llvm

if [ "$1" == "-h" ] || [ "$1" == "help" ] || [ "$1" == "-help" ] ; then
  echo " "
  echo "Example commands and actions: "
  echo "  ./build_emissary_mpi.sh                   cmake, make, NO Install "
  echo "  ./build_emissary_mpi.sh nocmake           NO cmake, make,  NO install "
  echo "  ./build_emissary_mpi.sh install           NO Cmake, make install "
  echo " "
  exit
fi

if [ ! -d "$EMISSARY_REPO_DIR" ] ; then
   echo "ERROR:  Missing repository $EMISSARY_REPO_DIR/"
   exit 1
fi

if [ ! -f "$LLVM_INSTALL_LOC/bin/clang" ] ; then
   echo "ERROR:  Missing file $LLVM_INSTALL_LOC/bin/clang"
   echo "        Build the AOMP llvm compiler in $AOMP first"
   echo "        This is needed to build the emissary libraries"
   echo " "
   exit 1
fi

check_writable_installdir "$1" "$INSTALL_EMISSARY"

export OPENMPI_DIR=$HOME/local/rocmopenmpi

patchrepo "$AOMP_REPOS/emissary"

if [ "$1" != "nocmake" ] && [ "$1" != "install" ] ; then
  if [ -d "$BUILD_EMISSARY_DIR" ] ; then
     echo
     echo "FRESH START , CLEANING UP FROM PREVIOUS BUILD"
     echo "rm -rf $BUILD_EMISSARY_DIR"
     rm -rf "$BUILD_EMISSARY_DIR"
  fi

  declare -a MYCMAKEOPTS

#  MYCMAKEOPTS=("${AOMP_ORIGIN_RPATH[@]}" -DCMAKE_BUILD_TYPE="$BUILDTYPE"
#               -DCMAKE_INSTALL_PREFIX="$INSTALL_EMISSARY"
#               -DCMAKE_BUILD_WITH_INSTALL_RPATH=ON
#               -DCMAKE_INSTALL_RPATH="\$ORIGIN/../lib"
#               -DCMAKE_EXE_LINKER_FLAGS='-Wl,--disable-new-dtags')

  MYCMAKEOPTS=(
    -DCMAKE_BUILD_TYPE="$BUILDTYPE"
    "${extra_cmake_opts[@]}"
    "-DLLVM_MAIN_SRC_DIR=$AOMP_REPOS/llvm-project"
    -DCMAKE_INSTALL_PREFIX="$INSTALL_EMISSARY")


  mkdir -p "$BUILD_EMISSARY_DIR"
  cd "$BUILD_EMISSARY_DIR" || exit
  echo
  echo " ---- Running ${AOMP_CMAKE} for emissary ---- "
  echo "${AOMP_CMAKE}" -B . "${MYCMAKEOPTS[@]}" -S "$EMISSARY_REPO_DIR"
  
  if ! ${AOMP_CMAKE} -B . "${MYCMAKEOPTS[@]}" -S "$EMISSARY_REPO_DIR" ; then
      echo "ERROR emissary cmake failed. Cmake flags"
      echo "      $(shquot "${MYCMAKEOPTS[@]}")"
      exit 1
  fi
fi

if [ "$1" = "cmake" ]; then
   exit 0
fi

cd "$BUILD_EMISSARY_DIR" || exit
echo
echo " ---- Running $AOMP_CMAKE -j $AOMP_JOB_THREADS for emissary ---- "

if ! ${AOMP_CMAKE} --build . --target all -j "$AOMP_JOB_THREADS" ; then
      echo " "
      echo "ERROR: ${AOMP_CMAKE} -j $AOMP_JOB_THREADS  FAILED"
      echo "To restart:"
      echo "  cd $BUILD_EMISSARY_DIR"
      echo "  $AOMP_CMAKE "
      exit 1
else
  if [ "$1" != "install" ] ; then
      echo
      echo " BUILD COMPLETE! To install emissary component run this command:"
      echo "  $0 install"
      echo
  fi
fi

#  ----------- Install only if asked  ----------------------------
if [ "$1" == "install" ] ; then
   cd "$BUILD_EMISSARY_DIR" || exit
   echo
   echo " -----Installing to $INSTALL_EMISSARY ----- "

   if ! $SUDO make install; then
      echo "ERROR make install failed "
      exit 1
   fi
   removepatch "$AOMP_REPOS/emissary"
fi

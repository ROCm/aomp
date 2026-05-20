#!/bin/bash
#
#  File: build_xio.sh
#        Build the rocm-xio library librocm-xio.a
#        The install option will install components into the aomp installation. 
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

XIO_REPO_DIR=$AOMP_REPOS/rocm-xio
echo "INFO: Getting latest sources for rocm-xio in dir \"$XIO_REPO_DIR\""
if [ -d "$XIO_REPO_DIR" ] ; then
  echo cd "$XIO_REPO_DIR"
  cd "$XIO_REPO_DIR" || exit
  echo git pull https://github.com/ROCm/rocm-xio.git
  git pull https://github.com/ROCm/rocm-xio.git
else 
  echo cd "$AOMP_REPOS"
  cd "$AOMP_REPOS" || exit
  echo git clone https://github.com/ROCm/rocm-xio.git
  git clone https://github.com/ROCm/rocm-xio.git
  cd "$XIO_REPO_DIR" || exit
fi

BUILD_DIR=${BUILD_AOMP}

BUILDTYPE="Release"

# Install XIO in the compiler directory of ROCm
INSTALL_XIO=${INSTALL_XIO:-$AOMP_INSTALL_DIR}/lib/llvm

if [ "$1" == "-h" ] || [ "$1" == "help" ] || [ "$1" == "-help" ] ; then
  echo " "
  echo "Example commands and actions: "
  echo "  ./build_xio.sh                   cmake, make, NO Install "
  echo "  ./build_xio.sh nocmake           NO cmake, make,  NO install "
  echo "  ./build_xio.sh install           NO Cmake, make install "
  echo " "
  exit
fi

if [ ! -d "$XIO_REPO_DIR" ] ; then
   echo "ERROR:  Missing repository $XIO_REPO_DIR/"
   exit 1
fi

if [ ! -f "$LLVM_INSTALL_LOC/bin/clang" ] ; then
   echo "ERROR:  Missing file $LLVM_INSTALL_LOC/bin/clang"
   echo "        Build the AOMP llvm compiler in $AOMP first"
   echo "        This is needed to build the xio libraries"
   echo " "
   exit 1
fi

check_writable_installdir "$1" "$INSTALL_XIO"

patchrepo "$AOMP_REPOS/rocm-xio"

if [ "$1" != "nocmake" ] && [ "$1" != "install" ] ; then
  if [ -d "$BUILD_DIR/build/rocm-xio" ] ; then
     echo
     echo "FRESH START , CLEANING UP FROM PREVIOUS BUILD"
     echo "rm -rf $BUILD_DIR/build/rocm-xio"
     rm -rf "$BUILD_DIR/build/rocm-xio"
  fi

  declare -a MYCMAKEOPTS

  MYCMAKEOPTS=("${AOMP_ORIGIN_RPATH[@]}" -DCMAKE_BUILD_TYPE="$BUILDTYPE"
               -DCMAKE_INSTALL_PREFIX="$INSTALL_XIO"
               -DCMAKE_BUILD_WITH_INSTALL_RPATH=ON
               -DCMAKE_INSTALL_RPATH="\$ORIGIN/../lib"
               -DCMAKE_EXE_LINKER_FLAGS='-Wl,--disable-new-dtags')

  mkdir -p "$BUILD_DIR/build/rocm-xio"
  cd "$BUILD_DIR/build/rocm-xio" || exit
  echo
  echo " -----Running ${AOMP_CMAKE} for xio ---- "
  echo "${AOMP_CMAKE} -B . $(shquot "${MYCMAKEOPTS[@]}") -S $XIO_REPO_DIR"
  
  if ! ${AOMP_CMAKE} -B . "${MYCMAKEOPTS[@]}" -S "$XIO_REPO_DIR" ; then
      echo "ERROR xio cmake failed. Cmake flags"
      echo "      $(shquot "${MYCMAKEOPTS[@]}")"
      exit 1
  fi
fi

if [ "$1" = "cmake" ]; then
   exit 0
fi

cd "$BUILD_DIR/build/rocm-xio" || exit
echo
echo " -----Running $AOMP_CMAKE -j $AOMP_JOB_THREADS for xio ---- "

if ! ${AOMP_CMAKE} --build . --target all -j "$AOMP_JOB_THREADS" ; then
      echo " "
      echo "ERROR: ${AOMP_CMAKE} -j $AOMP_JOB_THREADS  FAILED"
      echo "To restart:"
      echo "  cd $BUILD_DIR/build/rocm-xio"
      echo "  $AOMP_CMAKE "
      exit 1
else
  if [ "$1" != "install" ] ; then
      echo
      echo " BUILD COMPLETE! To install xio component run this command:"
      echo "  $0 install"
      echo
  fi
fi

#  ----------- Install only if asked  ----------------------------
if [ "$1" == "install" ] ; then
   cd "$BUILD_DIR/build/rocm-xio" || exit
   echo
   echo " -----Installing to $INSTALL_XIO ----- "

   if ! $SUDO make install; then
      echo "ERROR make install failed "
      exit 1
   fi
   removepatch "$AOMP_REPOS/rocm-xio"
fi

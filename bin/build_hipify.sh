#!/bin/bash
# MIT License
#
# Copyright (c) 2019 Advanced Micro Devices, Inc. All Rights Reserved.
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

HIPIFY_REPO_DIR=$AOMP_REPOS/hipify

BUILD_DIR=${BUILD_AOMP}

BUILDTYPE="Release"

INSTALL_HIPIFY=${INSTALL_HIPIFY:-$AOMP_INSTALL_DIR}

if [ "$1" == "-h" ] || [ "$1" == "help" ] || [ "$1" == "-help" ] ; then
  echo " "
  echo "Example commands and actions: "
  echo "  ./build_hipify.sh                   cmake, make, NO Install "
  echo "  ./build_hipify.sh nocmake           NO cmake, make,  NO install "
  echo "  ./build_hipify.sh install           NO Cmake, make install "
  echo " "
  exit
fi

if [ ! -d "$HIPIFY_REPO_DIR" ] ; then
   echo "ERROR:  Missing repository $HIPIFY_REPO_DIR/"
   exit 1
fi

if [ ! -f "$LLVM_INSTALL_LOC"/bin/clang ] ; then
   echo "ERROR:  Missing file $LLVM_INSTALL_LOC/bin/clang"
   echo "        Build and install the AOMP clang compiler in $AOMP first"
   echo "        This is needed to build hipify "
   echo " "
   exit 1
fi

# Make sure we can update the install directory
if [ "$1" == "install" ] ; then
   $SUDO mkdir -p "$INSTALL_HIPIFY"
   if ! $SUDO touch "$INSTALL_HIPIFY"/testfile; then
      echo "ERROR: No update access to $INSTALL_HIPIFY"
      exit 1
   fi
   $SUDO rm "$INSTALL_HIPIFY"/testfile
fi

if [ "$1" != "nocmake" ] && [ "$1" != "install" ] ; then

  if [ -d "$BUILD_DIR"/build/hipify ] ; then
     echo
     echo "FRESH START , CLEANING UP FROM PREVIOUS BUILD"
     echo "rm -rf $BUILD_DIR/build/hipify"
     rm -rf "$BUILD_DIR"/build/hipify
  fi

  declare -a MYCMAKEOPTS

  MYCMAKEOPTS=(-DCMAKE_BUILD_TYPE="$BUILDTYPE"
	       -DCMAKE_INSTALL_PREFIX="$AOMP_INSTALL_DIR"
	       -DCMAKE_PREFIX_PATH="$LLVM_INSTALL_LOC"
	       -DHIPIFY_INSTALL_CLANG_HEADERS=OFF
	       -DLLVM_EXTERNAL_LIT="$LLVM_INSTALL_LOC/bin/llvm-lit")

  mkdir -p "$BUILD_DIR"/build/hipify
  cd "$BUILD_DIR"/build/hipify || exit

  echo
  echo " -----Running cmake ---- "
  echo "${AOMP_CMAKE}" "${MYCMAKEOPTS[@]}" "$HIPIFY_REPO_DIR"
  if ! ${AOMP_CMAKE} "${MYCMAKEOPTS[@]}" "$HIPIFY_REPO_DIR"; then
      echo "ERROR hipify cmake failed. Cmake flags"
      echo "      $(shquote "${MYCMAKEOPTS[@]}")"
      exit 1
  fi
fi

if [ "$1" = "cmake" ]; then
  exit 0
fi

cd "$BUILD_DIR"/build/hipify || exit
echo
echo " -----Running make for hipify ---- "
if ! make -j "$AOMP_JOB_THREADS"; then
      echo " "
      echo "ERROR: make -j $AOMP_JOB_THREADS  FAILED"
      echo "To restart:"
      echo "  cd $BUILD_DIR/build/hipify"
      echo "  make "
      exit 1
else
  if [ "$1" != "install" ] ; then
      echo
      echo " BUILD COMPLETE! To install hipify component run this command:"
      echo "  $0 install"
      echo
  fi
fi

#  ----------- Install only if asked  ----------------------------
if [ "$1" == "install" ] ; then
      cd "$BUILD_DIR"/build/hipify || exit
      echo
      echo " -----Installing to $INSTALL_HIPIFY ----- "
      if ! $SUDO make install; then
         echo "ERROR make install failed "
         exit 1
      fi
fi

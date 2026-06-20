#!/bin/bash
#
#  File: build_hipfort.sh
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

# This is sourced from build_srock.sh so no header needed. 

REPO_DIR=$SROCK_REPOS/hipfort
BUILD_DIR=${REPO_DIR}/build
LLVM_INSTALL_LOC=$SROCK_INSTALL_DIR/lib/llvm
export HIP_PLATFORM=amd

if [ ! -f "$SROCK_CMAKE" ] ; then 
   echo "ERROR:  $0 requires SROCK_CMAKE env be set to qualified cmake"
   echo "        Did you source prebuild_srock.sh ? "
fi
if [ ! -d "$REPO_DIR" ] ; then
   echo "ERROR:  Missing repository $REPO_DIR/"
   exit 1
fi

if [ ! -f "$SROCK_INSTALL_DIR/lib/llvm/bin/clang" ] ; then
   echo "ERROR:  Missing file $SROCK_INSTALL_DIR/lib/llvm/bin/clang"
   echo " "
   exit 1
fi

if [ -d "$BUILD_DIR" ] ; then
   # shellcheck disable=SC2154 # $_build_srock_mode is set externally
   if [ "$_build_srock_mode" == "restart" ] ; then
      echo "===== Skipping $0"
      return
   fi
   echo
   echo "===== hipfort FRESH START, CLEANING UP FROM PREVIOUS BUILD"
   echo "      rm -rf $BUILD_DIR"
   rm -rf "$BUILD_DIR"
fi
mkdir -p "$BUILD_DIR"

declare -a MYCMAKEOPTS
MYCMAKEOPTS=(-DCMAKE_INSTALL_PREFIX="$SROCK_INSTALL_DIR/lib/llvm"
             -DCMAKE_BUILD_TYPE=Release
             -DCMAKE_Fortran_COMPILER="$LLVM_INSTALL_LOC/bin/flang"
             -DCMAKE_Fortran_FLAGS_DEBUG=""
             -DCMAKE_PREFIX_PATH="$SROCK_INSTALL_DIR/lib/cmake"
             -DCMAKE_AR="$LLVM_INSTALL_LOC/bin/llvm-ar"
             -DCMAKE_RANLIB="$LLVM_INSTALL_LOC/bin/llvm-ranlib")

cd "$REPO_DIR" || exit
echo
echo "===== Running $SROCK_CMAKE for hipfort"
$SROCK_CMAKE  -S. -Bbuild "${MYCMAKEOPTS[@]}" -DCMAKE_Fortran_FLAGS='-ffree-form -fPIC' "$REPO_DIR"
_cmakerc=$?
echo "===== DONE Running hipfort cmake"
if [ $_cmakerc != 0 ] ; then 
   echo
   echo "ERROR hipfort cmake failed with $SROCK_CMAKE. Cmake flags:"
   echo "      ${MYCMAKEOPTS[*]} -DCMAKE_Fortran_FLAGS='-ffree-form -fPIC'"
   exit 1
fi

echo
cd "$BUILD_DIR" || exit
echo "===== Running make -j16 for hipfort"
make -j16
_makerc=$?
if [ $_makerc != 0 ] ; then 
   echo " "
   echo "ERROR: make -j16 FAILED"
   echo "To restart:"
   echo "  cd $BUILD_DIR"
   echo "  make "
   exit 1
fi

echo
echo "===== Running make install"
make install 
_makerc=$?
if [ $_makerc != 0 ] ; then 
   echo "ERROR make install failed "
   exit 1
fi

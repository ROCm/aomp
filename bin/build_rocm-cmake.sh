#!/bin/bash
#
#  File: build_rocm-cmake.sh
#        Build the rocm-cmake component 
#        The install option will install component into the aomp installation. 
#
# MIT License
#
# Copyright (c) 2021 Advanced Micro Devices, Inc. All Rights Reserved.
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
realpath=`realpath $0`
thisdir=`dirname $realpath`
. $thisdir/aomp_common_vars
# --- end standard header ----

BUILD_DIR=${BUILD_AOMP}

BUILDTYPE="Release"

if [ "$1" == "-h" ] || [ "$1" == "help" ] || [ "$1" == "-help" ] ; then
  echo " "
  echo "Example commands and actions: "
  echo "  ./build_rocm-cmake.sh                   cmake, make, NO Install "
  echo "  ./build_rocm-cmake.sh install           NO Cmake, make install "
  echo " "
  exit
fi

echo checking for $AOMP_REPOS/$AOMP_ROCMCMAKE_REPO_NAME 
if [ ! -d $AOMP_REPOS/$AOMP_ROCMCMAKE_REPO_NAME ] ; then
   echo "ERROR:  Missing repository $AOMP_REPOS/$AOMP_ROCMCMAKE_REPO_NAME"
   exit 1
fi

# Make sure we can update the install directory
if [ "$1" == "install" ] ; then
   $SUDO mkdir -p $AOMP_INSTALL_DIR
   $SUDO touch $AOMP_INSTALL_DIR/testfile
   if [ $? != 0 ] ; then
      echo "ERROR: No update access to $AOMP_INSTALL_DIR"
      exit 1
   fi
   $SUDO rm $AOMP_INSTALL_DIR/testfile
fi

if [ "$1" != "nocmake" ] && [ "$1" != "install" ] ; then
  if [ -d "$BUILD_DIR/build/rocm-cmake" ] ; then
     echo
     echo "FRESH START , CLEANING UP FROM PREVIOUS BUILD"
     echo rm -rf $BUILD_DIR/build/rocm-cmake
     rm -rf $BUILD_DIR/build/rocm-cmake
  fi
 
  export CMAKE_PREFIX_PATH="""$AOMP_INSTALL_DIR"""
  MYCMAKEOPTS="-DCMAKE_INSTALL_PREFIX=$AOMP_INSTALL_DIR"

  mkdir -p $BUILD_DIR/build/rocm-cmake
  cd $BUILD_DIR/build/rocm-cmake
  echo
  echo " -----Running rocm-cmake cmake ---- "
  echo ${AOMP_CMAKE} $MYCMAKEOPTS $AOMP_REPOS/$AOMP_ROCMCMAKE_REPO_NAME
  ${AOMP_CMAKE} $MYCMAKEOPTS $AOMP_REPOS/$AOMP_ROCMCMAKE_REPO_NAME
  if [ $? != 0 ] ; then
      echo "ERROR rocm-cmake cmake failed. Cmake flags"
      echo "      $MYCMAKEOPTS"
      exit 1
  fi
fi

if [ "$1" = "cmake" ]; then
   exit 0
fi

cd $BUILD_DIR/build/rocm-cmake

#  ----------- no make for this component

#  ----------- Install only if asked  ----------------------------
if [ "$1" == "install" ] ; then
   cd $BUILD_DIR/build/rocm-cmake
   echo
   echo " -----Installing to $AOMP_INSTALL_DIR ----- "
   $SUDO make install
   if [ $? != 0 ] ; then
      echo "ERROR make install failed "
      exit 1
   fi
fi

#!/bin/bash
#
#  File: build_hip.sh
#        Build the hip host and device runtimes, 
#        The install option will install components into the aomp installation. 
#        The components include:
#          hip headers installed in $AOMP/include/hip
#          hip host runtime installed in $AOMP/lib/libhiprt.so
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

# --- Start standard header ----
function getdname(){
   local __DIRN=`dirname "$1"`
   if [ "$__DIRN" = "." ] ; then
      __DIRN=$PWD;
   else
      if [ ${__DIRN:0:1} != "/" ] ; then
         if [ ${__DIRN:0:2} == ".." ] ; then
               __DIRN=`dirname $PWD`/${__DIRN:3}
         else
            if [ ${__DIRN:0:1} = "." ] ; then
               __DIRN=$PWD/${__DIRN:2}
            else
               __DIRN=$PWD/$__DIRN
            fi
         fi
      fi
   fi
   echo $__DIRN
}
thisdir=$(getdname $0)
[ ! -L "$0" ] || thisdir=$(getdname `readlink "$0"`)
. $thisdir/aomp_common_vars
# --- end standard header ----

HIP_REPO_DIR=$AOMP_REPOS/$AOMP_HIP_REPO_NAME

PROC=`uname -p`
if [ "$PROC" == "ppc64le" ] ||  [ "$PROC" == "aarch64" ] ; then
   export HIP_PLATFORM="none"
else
   export HIP_PLATFORM="hcc"
fi

BUILD_DIR=${BUILD_AOMP}

BUILDTYPE="Release"

INSTALL_HIP=${INSTALL_HIP:-$AOMP_INSTALL_DIR}
LLVM_BUILD=$AOMP

REPO_BRANCH=$AOMP_HIP_REPO_BRANCH
REPO_DIR=$AOMP_REPOS/$AOMP_HIP_REPO_NAME
checkrepo

if [ "$1" == "-h" ] || [ "$1" == "help" ] || [ "$1" == "-help" ] ; then
  echo " "
  echo "Example commands and actions: "
  echo "  ./build_hip.sh                   cmake, make, NO Install "
  echo "  ./build_hip.sh nocmake           NO cmake, make,  NO install "
  echo "  ./build_hip.sh install           NO Cmake, make install "
  echo " "
  exit
fi

if [ ! -d $HIP_REPO_DIR ] ; then
   echo "ERROR:  Missing repository $HIP_REPO_DIR/"
   exit 1
fi

if [ ! -f $AOMP/bin/clang ] ; then
   echo "ERROR:  Missing file $AOMP/bin/clang"
   echo "        Build the AOMP llvm compiler in $AOMP first"
   echo "        This is needed to build the device libraries"
   echo " "
   exit 1
fi

# Make sure we can update the install directory
if [ "$1" == "install" ] ; then
   $SUDO mkdir -p $INSTALL_HIP
   $SUDO touch $INSTALL_HIP/testfile
   if [ $? != 0 ] ; then
      echo "ERROR: No update access to $INSTALL_HIP"
      exit 1
   fi
   $SUDO rm $INSTALL_HIP/testfile
fi

NUM_THREADS=
if [ ! -z `which "getconf"` ]; then
   NUM_THREADS=$(`which "getconf"` _NPROCESSORS_ONLN)
   if [ "$PROC" == "ppc64le" ] || [ "$PROC" == "aarch64" ] ; then
      NUM_THREADS=$(( NUM_THREADS / 2))
   fi

fi

if [ "$1" != "nocmake" ] && [ "$1" != "install" ] ; then

  if [ -d "$BUILD_DIR/build/hip" ] ; then
     echo
     echo "FRESH START , CLEANING UP FROM PREVIOUS BUILD"
     echo rm -rf $BUILD_DIR/build/hip
     rm -rf $BUILD_DIR/build/hip
  fi

  MYCMAKEOPTS="-DCMAKE_INSTALL_RPATH_USE_LINK_PATH=ON -DCMAKE_INSTALL_RPATH=$AOMP_INSTALL_DIR/lib:$AOMP_INSTALL_DIR/hcc/lib -DCMAKE_BUILD_TYPE=$BUILDTYPE -DCMAKE_INSTALL_PREFIX=$INSTALL_HIP -DHIP_PLATFORM=hcc -DHCC_HOME=$AOMP_INSTALL_DIR/hcc -DHSA_PATH=$AOMP_INSTALL_DIR/hsa -DHIP_INSTALL_ALL_SOURCE=OFF"
# -DBUILD_SHARED_LIBS=ON -DCMAKE_C_COMPILER=gcc -DCMAKE_CXX_COMPILER=g++ -DLLVM_DIR=$LLVM_BUILD/lib/cmake/llvm"

  mkdir -p $BUILD_DIR/build/hip
  cd $BUILD_DIR/build/hip
  echo
  echo " -----Running hip cmake ---- "
  echo cmake $MYCMAKEOPTS $HIP_REPO_DIR
  cmake $MYCMAKEOPTS $HIP_REPO_DIR
  if [ $? != 0 ] ; then
      echo "ERROR hip cmake failed. Cmake flags"
      echo "      $MYCMAKEOPTS"
      exit 1
  fi

fi

cd $BUILD_DIR/build/hip
echo
echo " -----Running make for hip ---- "
make -j $NUM_THREADS 
if [ $? != 0 ] ; then
      echo " "
      echo "ERROR: make -j $NUM_THREADS  FAILED"
      echo "To restart:"
      echo "  cd $BUILD_DIR/build/hip"
      echo "  make "
      exit 1
fi

#  ----------- Install only if asked  ----------------------------
if [ "$1" == "install" ] ; then
      cd $BUILD_DIR/build/hip
      echo
      echo " -----Installing to $INSTALL_HIP ----- "
      $SUDO make install
      if [ $? != 0 ] ; then
         echo "ERROR make install failed "
         exit 1
      fi
fi

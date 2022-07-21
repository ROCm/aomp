#!/bin/bash
#
#  File: build_vdi.sh
#        Build the vdi runtime
#        The install option will install components into the aomp installation. 
#
# MIT License
#
# Copyright (c) 2020 Advanced Micro Devices, Inc. All Rights Reserved.
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
  echo "  ./build_vdi.sh                   cmake, make, NO Install "
  echo " "
  exit
fi

if [ ! -d $AOMP_REPOS/$AOMP_VDI_REPO_NAME ] ; then
   echo "ERROR:  Missing repository $AOMP_REPOS/$AOMP_VDI_REPO_NAME"
   exit 1
fi
if [ ! -d $AOMP_REPOS/$AOMP_OCL_REPO_NAME ] ; then
   echo "ERROR:  Missing repository $AOMP_REPOS/$AOMP_OCL_REPO_NAME"
   exit 1
fi

GCCMIN=8
function getgcc8orless(){
   _loc=`which gcc`
   [ "$_loc" == "" ] && return
   gccver=`$_loc --version | grep gcc | cut -d")" -f2 | cut -d"." -f1`
   [ $gccver -gt $GCCMIN ] && _loc=`which gcc-$GCCMIN`
   echo $_loc
}
function getgxx8orless(){
   _loc=`which g++`
   [ "$_loc" == "" ] && return
   gxxver=`$_loc --version | grep g++ | cut -d")" -f2 | cut -d"." -f1`
   [ $gxxver -gt $GCCMIN ] && _loc=`which g++-$GCCMIN`
   echo $_loc
}

if [ "$AOMP_PROC" == "ppc64le" ] ; then
   GCCLOC=$(getgcc8orless)
   GXXLOC=$(getgxx8orless)
   if [ "$GCCLOC" == "" ] ; then
      echo "ERROR: NO ADEQUATE gcc"
      echo "       Please install gcc-8"
      exit 1
   fi
   if [ "$GXXLOC" == "" ] ; then
      echo "ERROR: NO ADEQUATE g++"
      echo "       Please install g++-8"
      exit 1
   fi
   COMPILERS="-DCMAKE_C_COMPILER=/usr/bin/gcc-8 -DCMAKE_CXX_COMPILER=/usr/bin/g++-8 -DCMAKE_CXX_FLAGS='-DNO_WARN_X86_INTRINSICS' -DCMAKE_C_FLAGS='-DNO_WARN_X86_INTRINSICS' "
else
   COMPILERS=""
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

patchrepo $AOMP_REPOS/$AOMP_VDI_REPO_NAME

if [ "$1" != "nocmake" ] && [ "$1" != "install" ] ; then
  if [ -d "$BUILD_DIR/build/vdi" ] ; then
     echo
     echo "FRESH START , CLEANING UP FROM PREVIOUS BUILD"
     echo rm -rf $BUILD_DIR/build/vdi
     rm -rf $BUILD_DIR/build/vdi
  fi

 # VDI needs opencl headers but vdi is built before opencl because opencl needs vdi.
 # The vdi cmake file hardcodes finding opencl headers in /opt/rocm/opencl but We don't want
 # to find them there for standalone builds that may not install rocm opencl.
 # We set OPENCL_INCLUDE_DIR to find the headers from the AOMP opencl repository.

  if [ $AOMP_STANDALONE_BUILD == 1 ] ; then
    echo lib_includes:$lib_includes
    MYCMAKEOPTS="$AOMP_ORIGIN_RPATH -DCMAKE_BUILD_TYPE=$BUILDTYPE -DCMAKE_INSTALL_PREFIX=$AOMP_INSTALL_DIR \
    -DUSE_COMGR_LIBRARY=yes \
    -DOPENCL_DIR=$AOMP_REPOS/$AOMP_OCL_REPO_NAME \
    -DCMAKE_CXX_FLAGS='-DCL_TARGET_OPENCL_VERSION=220' \
    -DCMAKE_MODULE_PATH=$ROCM_DIR/cmake \
    -DCMAKE_PREFIX_PATH=$ROCM_DIR/include;$ROCM_DIR/lib $COMPILERS"
  else
    MYCMAKEOPTS="$AOMP_ORIGIN_RPATH -DCMAKE_BUILD_TYPE=$BUILDTYPE -DCMAKE_INSTALL_PREFIX=$AOMP_INSTALL_DIR \
    -DUSE_COMGR_LIBRARY=yes \
    -DOPENCL_DIR=$AOMP_REPOS/$AOMP_OCL_REPO_NAME \
    -DCMAKE_CXX_FLAGS='-DCL_TARGET_OPENCL_VERSION=220' \
    -DCMAKE_MODULE_PATH=$ROCM_DIR/cmake \
    -DCMAKE_PREFIX_PATH=$ROCM_DIR/include;$ROCM_DIR;$THUNK_ROOT/include "
  fi

  mkdir -p $BUILD_DIR/build/vdi
  cd $BUILD_DIR/build/vdi
  echo
  echo " -----Running vdi cmake ---- "
  echo ${AOMP_CMAKE} $MYCMAKEOPTS $AOMP_REPOS/$AOMP_VDI_REPO_NAME
  ${AOMP_CMAKE} $MYCMAKEOPTS $AOMP_REPOS/$AOMP_VDI_REPO_NAME
  if [ $? != 0 ] ; then
      echo "ERROR vdi cmake failed. Cmake flags"
      echo "      $MYCMAKEOPTS"
      exit 1
  fi

fi

cd $BUILD_DIR/build/vdi
echo
echo " -----Running make for vdi ---- "
make -j $AOMP_JOB_THREADS 
if [ $? != 0 ] ; then
      echo " "
      echo "ERROR: make -j $AOMP_JOB_THREADS  FAILED"
      echo "To restart:"
      echo "  cd $BUILD_DIR/build/vdi"
      echo "  make "
      exit 1
else
  if [ "$1" != "install" ] ; then
      echo
      echo " BUILD COMPLETE! To install vdi  component run this command:"
      echo "  $0 install"

      echo
  fi
fi

if [ "$1" == "install" ] ; then
   cd $BUILD_DIR/build/vdi
   echo
   echo " -----Installing to $AOMP_INSTALL_DIR ----- "
   $SUDO make install
   if [ $? != 0 ] ; then
      echo "ERROR make install failed for vdi "
      exit 1
   fi
   removepatch $AOMP_REPOS/$AOMP_VDI_REPO_NAME
fi

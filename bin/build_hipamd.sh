#!/bin/bash
#
#  File: build_hipamd.sh
#        Build hip from hipamd, hip, ROCclr, and ROCm-OpenCL-Runtime repos
#        The install option will install components into the aomp installation. 
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

export HIPAMD_DIR=$AOMP_REPOS/hipamd
export HIP_DIR=$AOMP_REPOS/hip
export ROCclr_DIR=$AOMP_REPOS/ROCclr
export OPENCL_DIR=$AOMP_REPOS/ROCm-OpenCL-Runtime
[[ ! -d $HIPAMD_DIR ]] && echo "ERROR:  Missing $HIPAMD_DIR" && exit 1
[[ ! -d $HIP_DIR ]]    && echo "ERROR:  Missing $HIP_DIR"    && exit 1
[[ ! -d $ROCclr_DIR ]] && echo "ERROR:  Missing $ROCclr_DIR" && exit 1
[[ ! -d $OPENCL_DIR ]] && echo "ERROR:  Missing $OPENCL_DIR" && exit 1

export HSA_PATH=$AOMP_INSTALL_DIR
export HIP_PATH=$AOMP_INSTALL_DIR
export ROCM_PATH=$AOMP_INSTALL_DIR
export HIP_CLANG_PATH=$AOMP_INSTALL_DIR/bin
export DEVICE_LIB_PATH=$AOMP_INSTALL_DIR/lib

BUILD_DIR=${BUILD_AOMP}
BUILDTYPE="Release"

if [ "$1" == "-h" ] || [ "$1" == "help" ] || [ "$1" == "-help" ] ; then
  echo " "
  echo "Example commands and actions: "
  echo "  ./build_hipamd.sh                   cmake, make, NO Install "
  echo "  ./build_hipamd.sh nocmake           NO cmake, make,  NO install "
  echo "  ./build_hipamd.sh install           NO Cmake, make install "
  echo " "
  exit
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

patchrepo $AOMP_REPOS/hipamd

if [ "$1" != "nocmake" ] && [ "$1" != "install" ] ; then

  if [ -d "$BUILD_DIR/build/hipamd" ] ; then
     echo
     echo "FRESH START , CLEANING UP FROM PREVIOUS BUILD"
     echo rm -rf $BUILD_DIR/build/hipamd
     rm -rf $BUILD_DIR/build/hipamd
  fi

  MYCMAKEOPTS="$AOMP_ORIGIN_RPATH -DCMAKE_BUILD_TYPE=$BUILDTYPE \
 -DCMAKE_PREFIX_PATH=$AOMP_INSTALL_DIR \
 -DCMAKE_INSTALL_PREFIX=$AOMP_INSTALL_DIR \
 -DHIP_COMMON_DIR=$HIP_DIR \
 -DAMD_OPENCL_PATH=$OPENCL_DIR \
 -DROCCLR_PATH=$ROCclr_DIR \
 -DROCM_PATH=$ROCM_PATH"

  echo mkdir -p $BUILD_DIR/build/hipamd
  mkdir -p $BUILD_DIR/build/hipamd
  echo cd $BUILD_DIR/build/hipamd
  cd $BUILD_DIR/build/hipamd
  echo
  echo " -----Running hipamd cmake ---- "
  echo ${AOMP_CMAKE} $MYCMAKEOPTS $HIPAMD_DIR
  ${AOMP_CMAKE} $MYCMAKEOPTS $HIPAMD_DIR
  if [ $? != 0 ] ; then
      echo "ERROR hipamd cmake failed. Cmake flags"
      echo "      $MYCMAKEOPTS"
      exit 1
  fi

fi

cd $BUILD_DIR/build/hipamd

echo
echo " -----Running make for hipamd ---- "
make -j $AOMP_JOB_THREADS 
if [ $? != 0 ] ; then
      echo " "
      echo "ERROR: make -j $AOMP_JOB_THREADS  FAILED"
      echo "To restart:"
      echo "  cd $BUILD_DIR/build/hipamd"
      echo "  make "
      exit 1
else
  if [ "$1" != "install" ] ; then
      echo
      echo " BUILD COMPLETE! To install hipamd component run this command:"
      echo "  $0 install"
      echo
  fi
fi

#  ----------- Install only if asked  ----------------------------
if [ "$1" == "install" ] ; then
   cd $BUILD_DIR/build/hipamd
   echo
   echo " -----Installing to $AOMP_INSTALL_DIR ----- "
   $SUDO make install
   if [ $? != 0 ] ; then
      echo "ERROR make install failed "
      exit 1
   fi
   removepatch $AOMP_REPOS/hipamd

fi

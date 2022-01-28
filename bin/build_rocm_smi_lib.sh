#!/bin/bash
#
#  File: build_rocm_smi_lib.sh
#        Build the rocm_smi_lib library 
#        The install option will install components into the compiler installation. 
#
# MIT License
#
# Copyright (c) 2022 Advanced Micro Devices, Inc. All Rights Reserved.
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

repo_dir=$AOMP_REPOS/rocm_smi_lib

BUILDTYPE="Release"

LLVM_BUILD=$AOMP

if [ "$1" == "-h" ] || [ "$1" == "help" ] || [ "$1" == "-help" ] ; then
  echo " "
  echo "Example commands and actions: "
  echo "  ./build_rocm_smi_lib.sh                   cmake, make, NO Install "
  echo "  ./build_rocm_smi_lib.sh nocmake           NO cmake, make,  NO install "
  echo "  ./build_rocm_smi_lib.sh install           NO Cmake, make install "
  echo " "
  exit
fi

if [ ! -d $repo_dir ] ; then
   echo "ERROR:  Missing repository $repo_dir/"
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
  if [ -d "$BUILD_AOMP/build/rocm_smi_lib" ] ; then
     echo
     echo "FRESH START , CLEANING UP FROM PREVIOUS BUILD"
     echo rm -rf $BUILD_AOMP/build/rocm_smi_lib
     rm -rf $BUILD_AOMP/build/rocm_smi_lib
  fi

MYCMAKEOPTS="$AOMP_ORIGIN_RPATH -DCMAKE_BUILD_TYPE=$BUILDTYPE -DCMAKE_INSTALL_PREFIX=$AOMP_INSTALL_DIR -DROCM_DIR=$ROCM_DIR $AOMP_ORIGIN_RPATH"

  mkdir -p $BUILD_AOMP/build/rocm_smi_lib
  cd $BUILD_AOMP/build/rocm_smi_lib
  echo
  echo " -----Running rocm_smi_lib cmake ---- "
  echo ${AOMP_CMAKE} $MYCMAKEOPTS $repo_dir
  ${AOMP_CMAKE} $MYCMAKEOPTS $repo_dir
  if [ $? != 0 ] ; then
      echo "ERROR rocm_smi_lib cmake failed. Cmake flags"
      echo "      $MYCMAKEOPTS"
      exit 1
  fi

fi

cd $BUILD_AOMP/build/rocm_smi_lib
echo
echo " -----Running make for rocm_smi_lib ---- "
make -j $AOMP_JOB_THREADS 
if [ $? != 0 ] ; then
      echo " "
      echo "ERROR: make -j $AOMP_JOB_THREADS  FAILED"
      echo "To restart:"
      echo "  cd $BUILD_AOMP/build/rocm_smi_lib"
      echo "  make "
      exit 1
else
  if [ "$1" != "install" ] ; then
      echo
      echo " BUILD COMPLETE! To install rocm_smi_lib component run this command:"
      echo "  $0 install"
      echo
  fi
fi

#  ----------- Install only if asked  ----------------------------
if [ "$1" == "install" ] ; then
      cd $BUILD_AOMP/build/rocm_smi_lib
      echo
      echo " -----Installing to $AOMP_INSTALL_DIR ----- "
      $SUDO make install
      if [ $? != 0 ] ; then
         echo "ERROR make install failed "
         exit 1
      fi
fi

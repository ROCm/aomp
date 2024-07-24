#!/bin/bash
#
#  File: build_rocminfo.sh
#        Build the rocminfo utility
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
realpath=`realpath $0`
thisdir=`dirname $realpath`
. $thisdir/aomp_common_vars
# --- end standard header ----

RINFO_REPO_DIR=$AOMP_REPOS/$AOMP_RINFO_REPO_NAME

BUILD_DIR=${BUILD_AOMP}

BUILDTYPE="Release"

INSTALL_RINFO=${INSTALL_RINFO:-$AOMP_INSTALL_DIR}
LLVM_BUILD=$AOMP

if [ "$1" == "-h" ] || [ "$1" == "help" ] || [ "$1" == "-help" ] ; then
  echo " "
  echo "Example commands and actions: "
  echo "  ./build_rocminfo.sh                   cmake, make, NO Install "
  echo "  ./build_rocminfo.sh nocmake           NO cmake, make,  NO install "
  echo "  ./build_rocminfo.sh install           NO Cmake, make install "
  echo " "
  exit
fi

if [ ! -d $RINFO_REPO_DIR ] ; then
   echo "ERROR:  Missing repository $RINFO_REPO_DIR/"
   exit 1
fi

if [ ! -f $LLVM_INSTALL_LOC/bin/clang ] ; then
   echo "ERROR:  Missing file $LLVM_INSTALL_LOC/bin/clang"
   echo "        Build the AOMP llvm compiler in $AOMP first"
   echo "        This is needed to build the device libraries"
   echo " "
   exit 1
fi

# Make sure we can update the install directory
if [ "$1" == "install" ] ; then
   $SUDO mkdir -p $INSTALL_RINFO
   $SUDO touch $INSTALL_RINFO/testfile
   if [ $? != 0 ] ; then
      echo "ERROR: No update access to $INSTALL_RINFO"
      exit 1
   fi
   $SUDO rm $INSTALL_RINFO/testfile
fi

patchrepo $AOMP_REPOS/$AOMP_RINFO_REPO_NAME

if [ "$1" != "nocmake" ] && [ "$1" != "install" ] ; then
  if [ -d "$BUILD_DIR/build/rocminfo" ] ; then
     echo
     echo "FRESH START , CLEANING UP FROM PREVIOUS BUILD"
     echo rm -rf $BUILD_DIR/build/rocminfo
     rm -rf $BUILD_DIR/build/rocminfo
  fi

  MYCMAKEOPTS="$AOMP_ORIGIN_RPATH -DCMAKE_BUILD_TYPE=$BUILDTYPE -DCMAKE_INSTALL_PREFIX=$INSTALL_RINFO -DROCRTST_BLD_TYPE=$BUILDTYPE -DCMAKE_BUILD_WITH_INSTALL_RPATH=ON -DCMAKE_INSTALL_RPATH='\$ORIGIN/../lib' -DCMAKE_EXE_LINKER_FLAGS='-Wl,--disable-new-dtags'"

  mkdir -p $BUILD_DIR/build/rocminfo
  cd $BUILD_DIR/build/rocminfo
  echo
  echo " -----Running rocminfo cmake ---- "
  echo ${AOMP_CMAKE} $MYCMAKEOPTS $RINFO_REPO_DIR
  ${AOMP_CMAKE} $MYCMAKEOPTS $RINFO_REPO_DIR
  if [ $? != 0 ] ; then
      echo "ERROR rocminfo cmake failed. Cmake flags"
      echo "      $MYCMAKEOPTS"
      exit 1
  fi

fi

cd $BUILD_DIR/build/rocminfo
echo
echo " -----Running make for rocminfo ---- "
make -j $AOMP_JOB_THREADS 
if [ $? != 0 ] ; then
      echo " "
      echo "ERROR: make -j $AOMP_JOB_THREADS  FAILED"
      echo "To restart:"
      echo "  cd $BUILD_DIR/build/rocminfo"
      echo "  make "
      exit 1
else
  if [ "$1" != "install" ] ; then
      echo
      echo " BUILD COMPLETE! To install rocminfo component run this command:"
      echo "  $0 install"
      echo
  fi
fi

#  ----------- Install only if asked  ----------------------------
if [ "$1" == "install" ] ; then
      cd $BUILD_DIR/build/rocminfo
      echo
      echo " -----Installing to $INSTALL_RINFO ----- "
      $SUDO make install
      if [ $? != 0 ] ; then
         echo "ERROR make install failed "
         exit 1
      fi
      removepatch $AOMP_REPOS/$AOMP_RINFO_REPO_NAME
fi

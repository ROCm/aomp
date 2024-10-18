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
realpath=`realpath $0`
thisdir=`dirname $realpath`
. $thisdir/aomp_common_vars
# --- end standard header ----

HIPCC_REPO_DIR=$AOMP_REPOS/$AOMP_PROJECT_REPO_NAME/amd/hipcc

BUILD_DIR=${BUILD_AOMP}

BUILDTYPE="Release"

INSTALL_HIPCC=${INSTALL_HIPCC:-$AOMP_INSTALL_DIR}

if [ "$1" == "-h" ] || [ "$1" == "help" ] || [ "$1" == "-help" ] ; then
  echo " "
  echo "Example commands and actions: "
  echo "  ./build_hipcc.sh                   cmake, make, NO Install "
  echo "  ./build_hipcc.sh nocmake           NO cmake, make,  NO install "
  echo "  ./build_hipcc.sh install           NO Cmake, make install "
  echo " "
  exit
fi

if [ ! -d $EXTRAS_REPO_DIR ] ; then
   echo "ERROR:  Missing repository $EXTRAS_REPO_DIR/"
   exit 1
fi

if [ ! -f $LLVM_INSTALL_LOC/bin/clang ] ; then
   echo "ERROR:  Missing file $LLVM_INSTALL_LOC/bin/clang"
   echo "        Build and install the AOMP clang compiler in $AOMP first"
   echo "        This is needed to build hipcc "
   echo " "
   exit 1
fi

# Make sure we can update the install directory
if [ "$1" == "install" ] ; then
   $SUDO mkdir -p $INSTALL_HIPCC
   $SUDO touch $INSTALL_HIPCC/testfile
   if [ $? != 0 ] ; then
      echo "ERROR: No update access to $INSTALL_HIPCC"
      exit 1
   fi
   $SUDO rm $INSTALL_HIPCC/testfile
fi

if [ "$1" != "nocmake" ] && [ "$1" != "install" ] ; then

  if [ -d "$BUILD_DIR/build/hipcc" ] ; then
     echo
     echo "FRESH START , CLEANING UP FROM PREVIOUS BUILD"
     echo rm -rf $BUILD_DIR/build/hipcc
     rm -rf $BUILD_DIR/build/hipcc
  fi

  MYCMAKEOPTS="-DCMAKE_BUILD_TYPE=$BUILDTYPE -DCMAKE_INSTALL_PREFIX=$AOMP_INSTALL_DIR"

  mkdir -p $BUILD_DIR/build/hipcc
  cd $BUILD_DIR/build/hipcc

  export SED_INSTALL_DIR
  echo
  echo " -----Running cmake ---- "
  echo ${AOMP_CMAKE} $MYCMAKEOPTS $HIPCC_REPO_DIR
  ${AOMP_CMAKE} $MYCMAKEOPTS $HIPCC_REPO_DIR 
  if [ $? != 0 ] ; then
      echo "ERROR hipcc cmake failed. Cmake flags"
      echo "      $MYCMAKEOPTS"
      exit 1
  fi

fi

cd $BUILD_DIR/build/hipcc
echo
echo " -----Running make for hipcc ---- "
make -j $AOMP_JOB_THREADS 
if [ $? != 0 ] ; then
      echo " "
      echo "ERROR: make -j $AOMP_JOB_THREADS  FAILED"
      echo "To restart:"
      echo "  cd $BUILD_DIR/build/hipcc"
      echo "  make "
      exit 1
else
  if [ "$1" != "install" ] ; then
      echo
      echo " BUILD COMPLETE! To install hipcc component run this command:"
      echo "  $0 install"
      echo
  fi
fi

#  ----------- Install only if asked  ----------------------------
if [ "$1" == "install" ] ; then
      cd $BUILD_DIR/build/hipcc
      echo
      echo " -----Installing to $INSTALL_HIPCC ----- "
      $SUDO make install
      if [ $? != 0 ] ; then
         echo "ERROR make install failed "
         exit 1
      fi
fi

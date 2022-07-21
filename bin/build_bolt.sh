#!/bin/bash
#
# File: build_bolt.sh
#       Install bolt libsinto the compiler installation. 
#
# MIT License
#
# Copyright (c) 2023 Advanced Micro Devices, Inc. All Rights Reserved.
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

REPO_DIR=$AOMP_REPOS/bolt
BUILD_DIR=${BUILD_AOMP}
BOLT_INSTALL_DIR=${BOLT_INSTALL_DIR:-$AOMP_INSTALL_DIR}

if [ "$1" == "-h" ] || [ "$1" == "help" ] || [ "$1" == "-help" ] ; then
  echo " "
  echo "Example commands and actions: "
  echo "  ./build_bolt.sh                   cmake, make, NO Install "
  echo "  ./build_bolt.sh nocmake           NO cmake, make,  NO install "
  echo "  ./build_bolt.sh install           NO Cmake, make install "
  echo " "
  exit
fi

if [ ! -d $REPO_DIR ] ; then
   echo "ERROR:  Missing repository $REPO_DIR/"
   echo "        Try these commands till bolt is part of the AOMP manifest:"
   echo
   echo "  cd $AOMP_REPOS"
   echo "  git clone https://github.com/pmodels/bolt"
   echo
   exit 1
fi

if [ ! -f $AOMP/bin/clang ] ; then
   echo "ERROR:  Missing file $AOMP/bin/clang"
   echo "        Build the AOMP llvm compiler in $AOMP first"
   echo "        This is needed to build the bolt libraries"
   echo " "
   exit 1
fi

# Make sure we can update the install directory
if [ "$1" == "install" ] ; then
   $SUDO mkdir -p $BOLT_INSTALL_DIR
   $SUDO touch $BOLT_INSTALL_DIR/testfile
   if [ $? != 0 ] ; then
      echo "ERROR: No update access to $BOLT_INSTALL_DIR"
      exit 1
   fi
   $SUDO rm $BOLT_INSTALL_DIR/testfile
fi

patchrepo $AOMP_REPOS/bolt

if [ "$1" != "nocmake" ] && [ "$1" != "install" ] ; then
  if [ -d "$BUILD_DIR/build/bolt" ] ; then
     echo
     echo "FRESH START , CLEANING UP FROM PREVIOUS BUILD"
     echo rm -rf $BUILD_DIR/build/bolt
     rm -rf $BUILD_DIR/build/bolt
  fi

MYCMAKEOPTS=" \
-DCMAKE_INSTALL_PREFIX=$BOLT_INSTALL_DIR \
$AOMP_ORIGIN_RPATH \
-DCMAKE_C_COMPILER=$AOMP_CC_COMPILER \
-DCMAKE_CXX_COMPILER=$AOMP_CXX_COMPILER \
-DOPENMP_TEST_C_COMPILER=$AOMP_CC_COMPILER \
-DOPENMP_TEST_CXX_COMPILER=$AOMP_CXX_COMPILER \
-DCMAKE_BUILD_TYPE=Release \
-DOPENMP_ENABLE_LIBOMPTARGET=OFF \
-DLIBOMP_HEADERS_INSTALL_PATH=include/bolt \
-DLIBOMP_INSTALL_ALIASES=OFF \
-DLIBOMP_USE_ARGOBOTS=on"

  mkdir -p $BUILD_DIR/build/bolt
  cd $BUILD_DIR/build/bolt
  echo
  echo " -----Running bolt cmake ---- "
  echo ${AOMP_CMAKE} $MYCMAKEOPTS $REPO_DIR
  ${AOMP_CMAKE} $MYCMAKEOPTS $REPO_DIR
  if [ $? != 0 ] ; then
      echo "ERROR bolt cmake failed. Cmake flags"
      echo "      $MYCMAKEOPTS"
      exit 1
  fi

fi

cd $BUILD_DIR/build/bolt
echo
echo " -----Running make for bolt ---- "
make -j $AOMP_JOB_THREADS 
if [ $? != 0 ] ; then
      echo " "
      echo "ERROR: make -j $AOMP_JOB_THREADS  FAILED"
      echo "To restart:"
      echo "  cd $BUILD_DIR/build/bolt"
      echo "  make "
      exit 1
else
  if [ "$1" != "install" ] ; then
      echo
      echo " BUILD COMPLETE! To install bolt component run this command:"
      echo "  $0 install"
      echo
  fi
fi

#  ----------- Install only if asked  ----------------------------
if [ "$1" == "install" ] ; then
      cd $BUILD_DIR/build/bolt
      echo
      echo " -----Installing to $BOLT_INSTALL_DIR ----- "
      $SUDO make install
      if [ $? != 0 ] ; then
         echo "ERROR make install failed "
         exit 1
      fi
      removepatch $AOMP_REPOS/bolt
fi

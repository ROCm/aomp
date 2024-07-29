#!/bin/bash
#
#  File: build_amdsmi.sh
#        Build the amdsmi utility
#        The install option will install components into the aomp installation. 
#
# MIT License
#
# Copyright (c) 2024 Advanced Micro Devices, Inc. All Rights Reserved.
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

RSMILIB_REPO_DIR=$AOMP_REPOS/amdsmi

BUILD_DIR=${BUILD_AOMP}

BUILDTYPE="Release"

if [ "$1" == "-h" ] || [ "$1" == "help" ] || [ "$1" == "-help" ] ; then
  echo " "
  echo "Example commands and actions: "
  echo "  ./build_amdsmi.sh                   cmake, make, NO Install "
  echo "  ./build_amdsmi.sh nocmake           NO cmake, make,  NO install "
  echo "  ./build_amdsmi.sh install           NO Cmake, make install "
  echo " "
  exit
fi

if [ ! -d $RSMILIB_REPO_DIR ] ; then
   echo "ERROR:  Missing repository $RSMILIB_REPO_DIR/"
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

patchrepo $AOMP_REPOS/amdsmi

if [ "$1" != "nocmake" ] && [ "$1" != "install" ] ; then
  if [ -d "$BUILD_DIR/build/amdsmi" ] ; then
     echo
     echo "FRESH START , CLEANING UP FROM PREVIOUS BUILD"
     echo rm -rf $BUILD_DIR/build/amdsmi
     rm -rf $BUILD_DIR/build/amdsmi
  fi

  MYCMAKEOPTS="$AOMP_ORIGIN_RPATH -DCMAKE_BUILD_TYPE=$BUILDTYPE -DCMAKE_INSTALL_PREFIX=$AOMP_INSTALL_DIR -DCMAKE_BUILD_WITH_INSTALL_RPATH=ON -DCMAKE_INSTALL_RPATH='\$ORIGIN/../lib' -DCMAKE_EXE_LINKER_FLAGS='-Wl,--disable-new-dtags'"

  mkdir -p $BUILD_DIR/build/amdsmi
  cd $BUILD_DIR/build/amdsmi
  echo
  echo " -----Running amdsmi cmake ---- "
  echo ${AOMP_CMAKE} $MYCMAKEOPTS $RSMILIB_REPO_DIR
  ${AOMP_CMAKE} $MYCMAKEOPTS $RSMILIB_REPO_DIR
  if [ $? != 0 ] ; then
      echo "ERROR amdsmi cmake failed. Cmake flags"
      echo "      $MYCMAKEOPTS"
      exit 1
  fi

fi

cd $BUILD_DIR/build/amdsmi
echo
echo " -----Running make for amdsmi ---- "
make -j $AOMP_JOB_THREADS 
if [ $? != 0 ] ; then
      echo " "
      echo "ERROR: make -j $AOMP_JOB_THREADS  FAILED"
      echo "To restart:"
      echo "  cd $BUILD_DIR/build/amdsmi"
      echo "  make "
      exit 1
else
  if [ "$1" != "install" ] ; then
      echo
      echo " BUILD COMPLETE! To install amdsmi component run this command:"
      echo "  $0 install"
      echo
  fi
fi

#  ----------- Install only if asked  ----------------------------
if [ "$1" == "install" ] ; then
      cd $BUILD_DIR/build/amdsmi
      echo
      echo " -----Installing to $AOMP_INSTALL_DIR ----- "
      $SUDO make install
      if [ $? != 0 ] ; then
         echo "ERROR make install failed "
         exit 1
      fi
      removepatch $AOMP_REPOS/amdsmi
fi

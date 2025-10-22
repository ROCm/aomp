#!/bin/bash
#
#Copyright © Advanced Micro Devices, Inc., or its affiliates.
#
#SPDX-License-Identifier:  MIT
# 
#  build_half.sh: script to build and install half
#
# --- Start standard header to set AOMP environment variables ----
realpath=`realpath $0`
thisdir=`dirname $realpath`
. $thisdir/../aomp_common_vars
# --- end standard header ----

_libname=half
_repo_dir=$AOMP_REPOS/rocmlibs/$_libname
patchrepo $_repo_dir

if [ $AOMP_STANDALONE_BUILD == 1 ] ; then 
   if [ ! -L $AOMP ] ; then 
     if [ -d $AOMP ] ; then 
        echo "ERROR: Directory $AOMP is a physical directory."
        echo "       It must be a symbolic link or not exist"
        exit 1
     fi
   fi
else
   echo "ERROR: $0 only valid for AOMP_STANDALONE_BUILD=1"
   exit 1
fi

if [ "$1" == "nocmake" ] ; then 
   echo "ERROR: nocmake is not an option for $0"
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

if [ "$1" != "install" ] ; then
   echo 
   echo "This is a FRESH START. ERASING any previous builds in $BUILD_DIR/build/rocmlibs/$_libname"
   echo "Use ""$0 install"" to avoid FRESH START."
   echo rm -rf $BUILD_DIR/build/rocmlibs/$_libname
   rm -rf $BUILD_DIR/build/rocmlibs/$_libname
   mkdir -p $BUILD_DIR/build/rocmlibs/$_libname
else
   if [ ! -d $BUILD_DIR/build/rocmlibs/$_libname ] ; then 
      echo "ERROR: The build directory $BUILD_DIR/build/rocmlibs/$_libname does not exist"
      echo "       run $0 without install option. "
      exit 1
   fi
fi

if [ "$1" != "install" ] ; then
   # Remember start directory to return on exit
   _curdir=$PWD
   echo
   echo " -----Running cmake ---"
   echo cd $AOMP_REPOS/build/rocmlibs/$_libname
   cd $AOMP_REPOS/build/rocmlibs/$_libname
   pwd
   MYCMAKEOPTS="-DCMAKE_INSTALL_PREFIX=$AOMP_INSTALL_DIR"
   echo " ----- Running $AOMP_CMAKE $MYCMAKEOPTS $_repo_dir -----"
   $AOMP_CMAKE $MYCMAKEOPTS $_repo_dir
   if [ $? != 0 ] ; then 
      echo "ERROR cmake failed."
      echo "       $MYCMAKEOPTS"
      cd $_curdir
      exit 1
   fi

   echo " ----- Running ${AOMP_CMAKE} --build . -j $AOMP_JOB_THREADS -----"
   ${AOMP_CMAKE} --build . -j $AOMP_JOB_THREADS
   if [ $? != 0 ] ; then
      echo "ERROR: ${AOMP_CMAKE} --build . -j $AOMP_JOB_THREADS FAILED"
      exit 1
   fi
fi

if [ "$1" == "install" ] ; then
   echo " ----- Installing to $AOMP_INSTALL_DIR ----- "
   cd $AOMP_REPOS/build/rocmlibs/$_libname
   make -j$AOMP_JOB_THREADS install
   if [ $? != 0 ] ; then
      echo "ERROR install to $AOMP_INSTALL_DIR failed "
      exit 1
   fi
   echo
   echo "SUCCESSFUL INSTALL to $AOMP_INSTALL_DIR"
   echo
   removepatch $_repo_dir
else 
   echo 
   echo "SUCCESSFUL BUILD, please run:  $0 install"
   echo "  to install into $AOMP_INSTALL_DIR"
   echo 
fi

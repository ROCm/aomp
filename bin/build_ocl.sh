#!/bin/bash
#
#  File: build_ocl.sh
#        Build the ocl runtime
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

BUILD_DIR=${BUILD_AOMP}

BUILDTYPE="Release"

REPO_DIR=$AOMP_REPOS/$AOMP_OCL_REPO_NAME

if [ "$1" == "-h" ] || [ "$1" == "help" ] || [ "$1" == "-help" ] ; then
  echo " "
  echo "Example commands and actions: "
  echo "  ./build_ocl.sh                   cmake, make, NO Install "
  echo "  ./build_ocl.sh nocmake           NO cmake, make,  NO install "
  echo "  ./build_ocl.sh install           NO Cmake, make install "
  echo " "
  exit
fi

if [ ! -d $AOMP_REPOS/$AOMP_OCL_REPO_NAME ] ; then
   echo "ERROR:  Missing repository $AOMP_REPOS/$AOMP_OCL_REPO_NAME"
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

patchrepo $AOMP_REPOS/$AOMP_OCL_REPO_NAME

if [ "$1" != "nocmake" ] && [ "$1" != "install" ] ; then
  if [ -d "$BUILD_DIR/build/ocl" ] ; then
     echo
     echo "FRESH START , CLEANING UP FROM PREVIOUS BUILD"
     echo rm -rf $BUILD_DIR/build/ocl
     rm -rf $BUILD_DIR/build/ocl
  fi
 
  VDI_ROOT=$AOMP_REPOS/$AOMP_VDI_REPO_NAME
  export CMAKE_PREFIX_PATH="""$AOMP_INSTALL_DIR"""
  MYCMAKEOPTS="$AOMP_ORIGIN_RPATH_NO_DTAGS -DOpenGL_GL_PREFERENCE=LEGACY -DCMAKE_BUILD_TYPE=$BUILDTYPE -DCMAKE_INSTALL_PREFIX=$AOMP_INSTALL_DIR -DROCclr_DIR=$VDI_ROOT -DLIBROCclr_STATIC_DIR=$BUILD_DIR/build/vdi -DCMAKE_MODULE_PATH=$VDI_ROOT/cmake/modules -DCMAKE_PREFIX_PATH="$CMAKE_PREFIX_PATH" -DCMAKE_EXE_LINKER_FLAGS='-Wl,--disable-new-dtags'"

  mkdir -p $BUILD_DIR/build/ocl
  cd $BUILD_DIR/build/ocl
  echo
  echo " -----Running ocl cmake ---- "
  echo ${AOMP_CMAKE} $MYCMAKEOPTS $AOMP_REPOS/$AOMP_OCL_REPO_NAME
  ${AOMP_CMAKE} $MYCMAKEOPTS $AOMP_REPOS/$AOMP_OCL_REPO_NAME
  if [ $? != 0 ] ; then
      echo "ERROR ocl cmake failed. Cmake flags"
      echo "      $MYCMAKEOPTS"
      exit 1
  fi

fi

cd $BUILD_DIR/build/ocl
echo
echo " -----Running make for ocl ---- "
make -j $AOMP_JOB_THREADS 
if [ $? != 0 ] ; then
      echo " "
      echo "ERROR: make -j $AOMP_JOB_THREADS  FAILED"
      echo "To restart:"
      echo "  cd $BUILD_DIR/build/ocl"
      echo "  make "
      exit 1
else
  if [ "$1" != "install" ] ; then
      echo
      echo " BUILD COMPLETE! To install ocl component run this command:"
      echo "  $0 install"
      echo
  fi
fi

#  ----------- Install only if asked  ----------------------------
if [ "$1" == "install" ] ; then
   cd $BUILD_DIR/build/ocl
   echo
   echo " -----Installing to $AOMP_INSTALL_DIR ----- "
   $SUDO make install
   if [ $? != 0 ] ; then
      echo "ERROR make install failed "
      exit 1
   fi
   removepatch $AOMP_REPOS/$AOMP_OCL_REPO_NAME
fi

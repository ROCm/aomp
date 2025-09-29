#!/bin/bash
# 
#  build_rccl.sh: Script to build and install rccl.
#                 This uses a slightly modified install.sh from rccl. 
#                 It has a dependency on rocm-core.
#

BUILD_TYPE=${BUILD_TYPE:-Release}

# --- Start standard header to set AOMP environment variables ----
realpath=`realpath $0`
thisdir=`dirname $realpath`
. $thisdir/../aomp_common_vars
# --- end standard header ----

_howcalled=${0##*/}
_shname=${_howcalled#build_*}  # strip off build_
_libname=${_shname%*.sh}       # strip off .sh to get component libname = rccl
_source_dir=$AOMP_REPOS/rocmlibs/$_libname

if [ "$AOMP_USE_NINJA" == 0 ] ; then
    _set_ninja_gen=""
else
    _set_ninja_gen="--time-trace"
fi

patchrepo $_source_dir

export CC=$LLVM_INSTALL_LOC/bin/clang
export CXX=$LLVM_INSTALL_LOC/bin/clang++
export FC=$LLVM_INSTALL_LOC/bin/flang
export ROCM_DIR=$AOMP_INSTALL_DIR
export ROCM_PATH=$AOMP_INSTALL_DIR
# rccl needs cmake 3.25, so put prereq cmake first in path
export PATH=$AOMP_SUPP/cmake/bin:$AOMP_INSTALL_DIR/bin:$PATH
export NUM_PROC=$AOMP_JOB_THREADS
export CXXFLAGS="-I$HOME/local/rocm-core/include"
export LDFLAGS="-fPIC"
export EXPLICIT_ROCM_VERSION=$(grep -E "[0-9]+\.[0-9]+\.[0-9]+" < $HOME/local/rocm-core/.info/version)

if [ "$AOMP_USE_CCACHE" != 0 ] ; then
   _ccache_bin=`which ccache`
   export CMAKE_CXX_COMPILER_LAUNCHER=$_ccache_bin
fi

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
  _nocmake_option="--nocmake"
else
  _nocmake_option=""
fi

if [ "$BUILD_TYPE" == "Release" ] ; then
  _buildtype_option=""
  _build_dir_option="release"
else
  _buildtype_option="--debug"
  _build_dir_option="debug"
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
   echo
   echo "This is a FRESH START. ERASING any previous builds in $BUILD_DIR/build/rocmlibs/$_libname"
   echo "Use ""$0 install"" to avoid FRESH START."
   echo rm -rf $BUILD_DIR/build/rocmlibs/$_libname
   rm -rf $BUILD_DIR/build/rocmlibs/$_libname
   mkdir -p $BUILD_DIR/build/rocmlibs/$_libname
else
   if [ ! -d $BUILD_DIR/build/rocmlibs/$_libname ] ; then
      echo "ERROR: The build directory $BUILD_DIR/build/rocmlibs/$_libname does not exist"
      echo "       run $0 without install and without nocmake option"
      exit 1
   fi
fi

# Remember start directory to return on exit
_curdir=$PWD

if [ "$1" != "install" ] ; then
   echo
   echo " -----Running cmake in install.sh ---"
   echo cd $AOMP_REPOS/build/rocmlibs/$_libname
   cd $AOMP_REPOS/build/rocmlibs/$_libname
   echo ${AOMP_CMAKE} --toolchain=toolchain-linux.cmake -DCMAKE_BUILD_TYPE=Release -DGPU_TARGETS="$ROCMLIBS_GFXLIST" -DCMAKE_INSTALL_PREFIX=$AOMP_INSTALL_DIR -DROCM_PATH=$AOMP_INSTALL_DIR -DCOLLTRACE=OFF -DNPKIT_FLAGS="" -DONLY_FUNCS="" -DEXPLICIT_ROCM_VERSION=$EXPLICIT_ROCM_VERSION $_sour_dir
   ${AOMP_CMAKE} --toolchain=toolchain-linux.cmake -DCMAKE_BUILD_TYPE=Release -DGPU_TARGETS="$ROCMLIBS_GFXLIST" -DCMAKE_INSTALL_PREFIX=$AOMP_INSTALL_DIR -DROCM_PATH=$AOMP_INSTALL_DIR -DCOLLTRACE=OFF -DNPKIT_FLAGS="" -DONLY_FUNCS="" -DEXPLICIT_ROCM_VERSION=$EXPLICIT_ROCM_VERSION $_source_dir
  if [ $? != 0 ] ; then
     echo "ERROR cmake failed."
     echo "       $MYCMAKEOPTS"
     cd $_curdir
     exit 1
  fi
  make -j$AOMP_JOB_THREADS

  if [ $? != 0 ] ; then
     echo "ERROR make -j $AOMP_JOB_THREADS failed"
     exit 1
  fi
fi

if [ "$1" == "install" ] ; then
   echo " ----- Installing to $AOMP_INSTALL_DIR ----- "
   echo cd $AOMP_REPOS/build/rocmlibs/$_libname
   cd $AOMP_REPOS/build/rocmlibs/$_libname
   make -j$AOMP_JOB_THREADS install
   if [ $? != 0 ] ; then
      echo "ERROR install to $AOMP_INSTALL_DIR failed "
      exit 1
   fi
   echo
   echo "SUCCESSFUL INSTALL to $AOMP_INSTALL_DIR"
   echo
   removepatch $_source_dir
else
   echo
   echo "SUCCESSFUL BUILD, please run:  $0 install"
   echo "  to install into $AOMP_INSTALL_DIR"
   echo
fi

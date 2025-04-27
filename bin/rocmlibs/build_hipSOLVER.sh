#!/bin/bash
# 
#  build_rocsolver.sh:  Script to build and install rocsolver library
#
#
BUILD_TYPE=${BUILD_TYPE:-Release}

# --- Start standard header to set AOMP environment variables ----
realpath=`realpath $0`
thisdir=`dirname $realpath`
. $thisdir/../aomp_common_vars
# --- end standard header ----

_source_dir=$AOMP_REPOS/rocmlibs/hipSOLVER
_curdir=$PWD
cd $_source_dir

patchrepo $_source_dir

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

# This does not follow AOMP build directories convention because install.sh 
# assumes build directory is a subdirectory of source_directory.
# Changes to install.sh to fix this would be difficult.
_build_dir=$_source_dir/build
if [ "$1" != "install" ] ; then
   echo 
   echo "This is a FRESH START. ERASING any previous builds in $_build_dir"
   echo "Use ""$0 install"" to avoid FRESH START."
   echo rm -rf $_build_dir
   rm -rf $_build_dir
   mkdir -p $_build_dir
else
   if [ ! -d $_build_dir ] ; then 
      echo "ERROR: The build directory $_build_dir does not exist"
      echo "       run $0 without install option. "
      exit 1
   fi
fi

cd $_source_dir
if [ "$1" != "install" ] ; then
   # Remember start directory to return on exit
   _curdir=$PWD
   echo " ----- Running hipSOLVER install.sh -----"
   export ROCM_PATH=$AOMP_INSTALL_DIR
   _cmd="./install.sh --compiler $AOMP_INSTALL_DIR/lib/llvm/bin/clang++ --rocblas-path $AOMP_INSTALL_DIR --hipblas-path $AOMP_INSTALL_DIR --rocsolver-path $AOMP_INSTALL_DIR --cmakepp $AOMP_INSTALL_DIR --no-sparse --no-hip-clang --relocatable --cmake-arg -DCMAKE_INSTALL_PREFIX=$AOMP_INSTALL_DIR"
   $_cmd
   if [ $? != 0 ] ; then 
      echo "ERROR $_cmd failed."
      cd $_curdir
      exit 1
   fi
fi

if [ "$1" == "install" ] ; then
   echo " -----Installing to $AOMP_INSTALL_DIR ---- "
   cd $_build_dir/release
   make install
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

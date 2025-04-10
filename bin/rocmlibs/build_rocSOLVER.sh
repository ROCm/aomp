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

_repo_dir=$AOMP_REPOS/rocmlibs/rocSOLVER
patchrepo $_repo_dir

export CC=$LLVM_INSTALL_LOC/bin/clang 
export CXX=$LLVM_INSTALL_LOC/bin/clang++
export ROCM_DIR=$AOMP_INSTALL_DIR
export ROCM_PATH=$AOMP_INSTALL_DIR
export PATH=$AOMP_SUPP/cmake/bin:$AOMP_INSTALL_DIR/bin:$PATH
export HIP_USE_PERL_SCRIPTS=1
export USE_PERL_SCRIPTS=1
#  Reduce number of jobs because of large mem requirements for linking
_num_procs=$(( $AOMP_JOB_THREADS / 2 ))
export NUM_PROC=$_num_procs
export CXXFLAGS="-I$AOMP_INSTALL_DIR/include -D__HIP_PLATFORM_AMD__=1" 
export LDFLAGS="-fPIC"
if [ "$AOMP_USE_CCACHE" != 0 ] ; then
   _ccache_bin=`which ccache`
  # export CMAKE_CXX_COMPILER_LAUNCHER=$_ccache_bin
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
   echo "This is a FRESH START. ERASING any previous builds in $BUILD_DIR/build/rocmlibs/rocSOLVER"
   echo "Use ""$0 install"" to avoid FRESH START."
   echo rm -rf $BUILD_DIR/build/rocmlibs/rocSOLVER
   rm -rf $BUILD_DIR/build/rocmlibs/rocSOLVER
   mkdir -p $BUILD_DIR/build/rocmlibs/rocSOLVER
else
   if [ ! -d $BUILD_DIR/build/rocmlibs/rocSOLVER ] ; then 
      echo "ERROR: The build directory $BUILD_DIR/build/rocmlibs/rocSOLVER does not exist"
      echo "       run $0 without install option. "
      exit 1
   fi
fi

if [ "$1" != "install" ] ; then
   # Remember start directory to return on exit
   _curdir=$PWD
   echo " -----Running cmake ---"
   echo cd $AOMP_REPOS/build/rocmlibs/rocSOLVER
   cd $AOMP_REPOS/build/rocmlibs/rocSOLVER
   pwd
   MYCMAKEOPTS="
     -DCMAKE_CXX_COMPILER=$CXX
     -DCMAKE_C_COMPILER=$CC
     -DROCM_DIR:PATH=$AOMP_INSTALL_DIR
     -DCPACK_PACKAGING_INSTALL_PREFIX=$AOMP_INSTALL_DIR
     -DCMAKE_INSTALL_PREFIX=$AOMP_INSTALL_DIR
     -DROCM_PATH=$AOMP_INSTALL_DIR
     -DCMAKE_PREFIX_PATH:PATH=$AOMP_INSTALL_DIR
     -DCPACK_SET_DESTDIR=OFF
     -DCMAKE_BUILD_TYPE=Release
     -Drocblas_DIR=$AOMP_INSTALL_DIR/rocblas
     -DAMDGPU_TARGETS="""$_gfxlist"""
   "
   echo $AOMP_CMAKE $MYCMAKEOPTS $_repo_dir
   $AOMP_CMAKE $MYCMAKEOPTS $_repo_dir
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
   echo " -----Installing to $AOMP_INSTALL_DIR ---- "
   cd $AOMP_REPOS/build/rocmlibs/rocSOLVER
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

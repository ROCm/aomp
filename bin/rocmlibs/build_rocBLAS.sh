#!/bin/bash
# 
#  build_rocblas.sh:  Script to build and install rocblas library
#
#
BUILD_TYPE=${BUILD_TYPE:-Release}

# --- Start standard header to set AOMP environment variables ----
realpath=`realpath $0`
thisdir=`dirname $realpath`
. $thisdir/../aomp_common_vars
# --- end standard header ----

_repo_dir=$AOMP_REPOS/rocmlibs/rocBLAS

# Check if Tensile is to be built with rocBLAS
AOMP_BUILD_TENSILE=${AOMP_BUILD_TENSILE:-1}
if [ $AOMP_BUILD_TENSILE == 0 ] ; then 
   echo 
   echo "WARNING: Building rocblas without Tensile"
   _local_tensile_opt="--no_tensile"
else
   _cwd=$PWD
   _tensile_repo_dir=$AOMP_REPOS/rocmlibs/Tensile
   cd $_tensile_repo_dir
   # Read the commit SHA from the file rocBLAS/tensile_tag.txt
   _tensile_commit_sha=$(cat $_repo_dir/tensile_tag.txt)
   # Checkout the specific commit SHA
   git checkout $_tensile_commit_sha
   echo "Checking out Tensile commit $_tensile_commit_sha"
   cd $_cwd
   _local_tensile_opt="--test_local_path=$_tensile_repo_dir"
   patchrepo $_tensile_repo_dir
fi

# Check if rocBLAS is to be built with hipBLASLT
# It won't work unless hipBLASLT is already installed
ROCBLAS_USE_HIPBLASLT=${ROCBLAS_USE_HIPBLASLT:-0}
if [ $ROCBLAS_USE_HIPBLASLT == 0 ] ; then
   echo
   echo "WARNING: Building rocblas without hipBLASLT"
   _local_hipblaslt_opt="--no_hipblaslt"
fi

patchrepo $_repo_dir

export CC=$LLVM_INSTALL_LOC/bin/amdclang
export CXX=$LLVM_INSTALL_LOC/bin/amdclang++
export FC=$LLVM_INSTALL_LOC/bin/amdflang
export ROCM_DIR=$AOMP_INSTALL_DIR
export ROCM_PATH=$AOMP_INSTALL_DIR
export PATH=$AOMP_SUPP/cmake/bin:$AOMP_INSTALL_DIR/bin:$AOMP/llvm/bin:$PATH
export HIP_USE_PERL_SCRIPTS=1
export USE_PERL_SCRIPTS=1
export CXXFLAGS="-I$AOMP_INSTALL_DIR/include -D__HIP_PLATFORM_AMD__=1"
export LDFLAGS="-fPIC"

## this causes fail when building with Tensile
#export TENSILE_SKIP_LIBRARY=1
if [ "$AOMP_USE_CCACHE" != 0 ] ; then
   _ccache_bin=`which ccache`
  # export CMAKE_CXX_COMPILER_LAUNCHER=$_ccache_bin
fi

# Set _build_type_option to Release or Debug based on BUILD_TYPE
if [ "$BUILD_TYPE" == "Debug" ] ; then
   _build_type_option="--debug"
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
   echo "ERROR: nocmake is not an option for $0 because we use rmake.py"
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
   echo "This is a FRESH START. ERASING any previous builds in $BUILD_DIR/build/rocmlibs/rocBLAS"
   echo "Use ""$0 install"" to avoid FRESH START."
   echo rm -rf $BUILD_DIR/build/rocmlibs/rocBLAS
   rm -rf $BUILD_DIR/build/rocmlibs/rocBLAS
   mkdir -p $BUILD_DIR/build/rocmlibs/rocBLAS
   if [ $AOMP_BUILD_TENSILE != 0 ] ; then 
      # Cleanup possible old tensile build area
      echo rm -rf $_tensile_repo_dir/build
      rm -rf $_tensile_repo_dir/build
   fi
else
   if [ ! -d $BUILD_DIR/build/rocmlibs/rocBLAS ] ; then
      echo "ERROR: The build directory $BUILD_DIR/build/rocmlibs/rocBLAS"
      echo "       run $0 without install option. "
      exit 1
   fi
fi

if [ "$1" != "install" ] ; then
   # Remember start directory to return on exit
   _curdir=$PWD
   MYCMAKEOPTS="
     -DCMAKE_TOOLCHAIN_FILE=toolchain-linux.cmake
     -DCMAKE_CXX_COMPILER=$CXX
     -DCMAKE_C_COMPILER=$CC
     -DROCM_DIR:PATH=$AOMP_INSTALL_DIR
     -DCPACK_PACKAGING_INSTALL_PREFIX=$AOMP_INSTALL_DIR
     -DCMAKE_INSTALL_PREFIX=$AOMP_INSTALL_DIR
     -DROCM_PATH=$AOMP_INSTALL_DIR
     -DCMAKE_PREFIX_PATH:PATH=$AOMP_INSTALL_DIR
     -DCPACK_SET_DESTDIR=OFF
     -DCMAKE_BUILD_TYPE=Release
     -DTensile_CODE_OBJECT_VERSION=default
     -DTensile_LOGIC=asm_full
     -DTensile_TEST_LOCAL_PATH=$AOMP_REPOS/rocmlibs/Tensile
     -DTensile_SEPARATE_ARCHITECTURES=ON
     -DTensile_LAZY_LIBRARY_LOADING=ON
     -DTensile_LIBRARY_FORMAT=msgpack
     -DBUILD_WITH_HIPBLASLT=OFF
     -DAMDGPU_TARGETS="""$_gfxlist"""
    "
   echo "Beginning cmake for rocblas..."
   cd $BUILD_DIR/build/rocmlibs/rocBLAS
   echo $AOMP_CMAKE $MYCMAKEOPTS $_repo_dir
   $AOMP_CMAKE $MYCMAKEOPTS $_repo_dir
   if [ $? != 0 ] ; then
     echo "ERROR cmake failed. Cmake flags"
     echo "      $MYCMAKEOPTS"
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

   if [ "$BUILD_TYPE" == "Release" ] ; then
      _build_type_dir=release
   else
      _build_type_dir=debug
   fi
   cd $BUILD_DIR/build/rocmlibs/rocBLAS
   make -j$AOMP_JOB_THREADS install
   if [ $? != 0 ] ; then
      echo "ERROR install to $AOMP_INSTALL_DIR failed "
      exit 1
   fi
   echo
   echo "SUCCESSFUL INSTALL to $AOMP_INSTALL_DIR"
   echo
   removepatch $_repo_dir
   removepatch $_tensile_repo_dir
else 
   echo 
   echo "SUCCESSFUL BUILD, please run:  $0 install"
   echo "  to install into $AOMP_INSTALL_DIR"
   echo 
fi

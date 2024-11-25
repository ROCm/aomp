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
_build_dir=$_repo_dir/build

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

_gfxlist=""
 _sep=""
for _arch in $GFXLIST ; do 
 if [ $_arch == "gfx90a" ] ; then 
     _gfxlist+="${_sep}gfx90a:xnack-"
     _gfxlist+=";gfx90a:xnack+"
 else
     _gfxlist+=${_sep}$_arch
 fi
 _sep=";"
done
export CC=$LLVM_INSTALL_LOC/bin/clang
export CXX=$LLVM_INSTALL_LOC/bin/clang++
export FC=$LLVM_INSTALL_LOC/bin/flang
export ROCM_DIR=$AOMP_INSTALL_DIR
export ROCM_PATH=$AOMP_INSTALL_DIR
export PATH=$AOMP_SUPP/cmake/bin:$AOMP_INSTALL_DIR/bin:$PATH
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
   echo "This is a FRESH START. ERASING any previous builds in $_build_dir"
   echo "Use ""$0 install"" to avoid FRESH START."
   echo rm -rf $_build_dir
   rm -rf $_build_dir
   mkdir -p $_build_dir
   if [ $AOMP_BUILD_TENSILE != 0 ] ; then 
      # Cleanup possible old tensile build area
      echo rm -rf $_tensile_repo_dir/build
      rm -rf $_tensile_repo_dir/build
   fi
else
   if [ ! -d $_build_dir ] ; then 
      echo "ERROR: The build directory $_build_dir"
      echo "       run $0 without install option. "
      exit 1
   fi
fi

if [ "$1" != "install" ] ; then
   # Remember start directory to return on exit
   _curdir=$PWD
   echo
   echo " ----- Running python3 rmake.py -----"
   # python rmake.py must be run from source directory.
   echo cd $_repo_dir
   cd $_repo_dir
   _rmake_py_cmd="python3 ./rmake.py \
$_local_tensile_opt \
$_local_hipblaslt_opt \
$_build_type_option \
--install_invoked \
--build_dir $_build_dir \
--src_path=$_repo_dir \
--jobs=$AOMP_JOB_THREADS \
--architecture="""$_gfxlist""" \
"

# other unused options for rmake.py
#--no-merge-architectures \
#--no-lazy-library-loading \

   echo 
   echo "$_rmake_py_cmd "
   echo 
   $_rmake_py_cmd 2>&1
   if [ $? != 0 ] ; then 
      echo "ERROR rmake.py failed."
      echo "       cmd:$_rmake_py_cmd"
      cd $_curdir
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
   echo rsync -av $_build_dir/$_build_type_dir/rocblas-install/ $AOMP_INSTALL_DIR/
   rsync -av $_build_dir/$_build_type_dir/rocblas-install/ $AOMP_INSTALL_DIR/
   if [ $? != 0 ] ; then
      echo "ERROR copy to $AOMP_INSTALL_DIR failed "
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

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

AOMP_BUILD_TENSILE=${AOMP_BUILD_TENSILE:-0}

if [ $AOMP_BUILD_TENSILE == 0 ] ; then 
   echo 
   echo "WARNING: Building rocblas without Tensile"
   _local_tensile_opt=""
else
   _tensile_repo_dir=$AOMP_REPOS/rocmlibs/Tensile
   _cwd=$PWD
   cd $_tensile_repo_dir
   git checkout release/rocm-rel-6.2
   git pull
   # FIXME:  We should get the Tensile hash from rocBLAS/tensile_tag.txt
   #git checkout 97e0cfc2c8cb87a1e38901d99c39090dc4181652
   git checkout 66ab453c3fcfc3f3816e3383cb0eccb528a1b5a9
   cd $_cwd
   _local_tensile_opt="--test_local_path=$_tensile_repo_dir"
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
export CC=$AOMP_INSTALL_DIR/bin/hipcc
export CXX=$AOMP_INSTALL_DIR/bin/hipcc
export FC=gfortran
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
--install_invoked \
--build_dir $_build_dir \
--src_path=$_repo_dir \
--no_tensile \
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
   echo rsync -av $_build_dir/release/rocblas-install/ $AOMP_INSTALL_DIR/
   rsync -av $_build_dir/release/rocblas-install/ $AOMP_INSTALL_DIR/
   if [ $? != 0 ] ; then
      echo "ERROR copy to $AOMP_INSTALL_DIR failed "
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

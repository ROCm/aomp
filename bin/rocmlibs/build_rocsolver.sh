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
export CC=$AOMP_INSTALL_DIR/bin/clang 
export CXX=$AOMP_INSTALL_DIR/bin/clang++
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
   echo
   echo " -----Running python3 rmake.py ---"
   # python rmake.py must be run from source directory.
   echo cd $_repo_dir
   cd $_repo_dir
   _cmd="python3 ./rmake.py \
--build_dir=$BUILD_DIR/build/rocmlibs/rocSOLVER \
--rocblas_dir=$AOMP_INSTALL_DIR/rocblas \
--install \
--architecture="\"$_gfxlist"\" \
"
   echo $_cmd
   $_cmd 2>&1
   if [ $? != 0 ] ; then 
      echo "ERROR rmake.py failed."
      echo "       cmd:$_cmd"
      cd $_curdir
      exit 1
   fi
fi

if [ "$1" == "install" ] ; then
   echo " -----Installing to $AOMP_INSTALL_DIR ---- "
   echo rsync -av $BUILD_DIR/build/rocmlibs/rocSOLVER/release/rocsolver-install/ $AOMP_INSTALL_DIR/
   rsync -av $BUILD_DIR/build/rocmlibs/rocSOLVER/release/rocsolver-install/ $AOMP_INSTALL_DIR/
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

#!/bin/bash
# 
#  build_powerinfer.sh:  Script to build and install powerinfer library
#                     This build is classic  cmake, make, make install
#
BUILD_TYPE=${BUILD_TYPE:-Release}

# --- Start standard header to set AOMP environment variables ----
realpath=`realpath $0`
thisdir=`dirname $realpath`
. $thisdir/../aomp_common_vars
# --- end standard header ----

# Patch rocr
_repo_dir=$AOMP_REPOS/rocmlibs/PowerInfer
patchrepo $_repo_dir

if [ "$AOMP_USE_NINJA" == 0 ] ; then
    AOMP_SET_NINJA_GEN=""
else
    AOMP_SET_NINJA_GEN="-G Ninja"
fi

GFXSEMICOLONS=`echo $GFXLIST | tr ' ' ';' `
GFXSEMICOLONS=""$GFXSEMICOLONS""
export CC=$AOMP/bin/clang 
export CXX=$AOMP/bin/clang++
#export ROCM_DIR=$AOMP
#export PATH=$AOMP_SUPP/cmake/bin:$AOMP/bin:$PATH
#export PATH=$AOMP/bin:$PATH
export USE_PERL_SCRIPTS=1
export NUM_PROC=$AOMP_JOB_THREADS
export AMDGPU_TARGETS="$GFXSEMICOLONS"
export CXXFLAGS="-I$AOMP_INSTALL_DIR/include -D__HIP_PLATFORM_AMD__=1"
#export LDFLAGS="-fPIC"

MYCMAKEOPTS="-DCMAKE_BUILD_TYPE=$BUILD_TYPE \
-DLLAMA_HIPBLAS=ON \
-DCMAKE_PREFIX_PATH=$AOMP_INSTALL_DIR/lib/cmake  \
-DCMAKE_INSTALL_PREFIX=$AOMP_INSTALL_DIR/PowerInfer \
-DCMAKE_C_COMPILER=$AOMP/bin/clang \
-DCMAKE_CXX_COMPILER=$AOMP/bin/clang++ \
-DHIP_PLATFORM=amd \
-DAMDGPU_TARGETS="\'$GFXSEMICOLONS\'" \
"
# $AOMP_ORIGIN_RPATH \
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
   echo "This is a FRESH START. ERASING any previous builds in $BUILD_DIR/build/rocmlibs/PowerInfer"
   echo "Use ""$0 nocmake"" or ""$0 install"" to avoid FRESH START."
   rm -rf $BUILD_DIR/build/rocmlibs/PowerInfer
   mkdir -p $BUILD_DIR/build/rocmlibs/PowerInfer
else
   if [ ! -d $BUILD_DIR/build/rocmlibs/PowerInfer ] ; then 
      echo "ERROR: The build directory $BUILD_DIR/build/rocmlibs/PowerInfer "
      echo "       run $0 without nocmake or install options. "
      exit 1
   fi
fi

cd $BUILD_DIR/build/rocmlibs/PowerInfer

if [ "$1" != "nocmake" ] && [ "$1" != "install" ] ; then
   echo
   echo " -----Running cmake ---- "
   echo cmake $MYCMAKEOPTS $_repo_dir
   env CXX=$AOMP/bin/hipcc cmake $MYCMAKEOPTS $_repo_dir  2>&1
   if [ $? != 0 ] ; then 
      echo "ERROR cmake failed. Cmake flags"
      echo "      $MYCMAKEOPTS"
      exit 1
   fi
fi

echo
echo " -----Running make ---- "
echo make -j $AOMP_JOB_THREADS
env CXX=$AOMP_INSTALL_DIR/bin/hipcc make -j $AOMP_JOB_THREADS
if [ $? != 0 ] ; then
   echo "ERROR make -j $AOMP_JOB_THREADS failed"
   exit 1
fi

if [ "$1" == "install" ] ; then
   echo " -----Installing to $AOMP_INSTALL_DIR ---- "
   $SUDO make install
   if [ $? != 0 ] ; then
      echo "ERROR make install install failed "
      exit 1
   fi
   echo
   echo "SUCCESSFUL INSTALL to $AOMP_INSTALL_DIR/PowerInfer"
   echo
   removepatch $_repo_dir
else 
   echo 
   echo "SUCCESSFUL BUILD, please run:  $0 install"
   echo "  to install into $AOMP_INSTALL_DIR/PowerInfer "
   echo 
fi

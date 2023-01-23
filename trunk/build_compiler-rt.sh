#!/bin/bash
# 
#  build_project.sh:  Script to build the trunk compiler. 
#
BUILD_TYPE=${BUILD_TYPE:-Release}

# --- Start standard header to set AOMP environment variables ----
realpath=`realpath $0`
thisdir=`dirname $realpath`
. $thisdir/trunk_common_vars
# --- end standard header ----

if [ "$AOMP_USE_NINJA" == 0 ] ; then
    _set_ninja_gan=""
else
    _set_ninja_gan="-G Ninja"
fi

export LLVM_DIR=$TRUNK_INSTALL_DIR

MYCMAKEOPTS="\
$_set_ninja_gan \
-DCMAKE_C_COMPILER=$TRUNK_INSTALL_DIR/bin/clang \
-DCMAKE_CXX_COMPILER=$TRUNK_INSTALL_DIR/bin/clang++ \
-DCMAKE_BUILD_TYPE=$BUILD_TYPE \
-DCMAKE_INSTALL_PREFIX=$TRUNK_INSTALL_DIR \
-DCMAKE_CXX_STANDARD=11 \
-DCMAKE_C_FLAGS=-mlong-double-128 \
-DCMAKE_CXX_FLAGS=-mlong-double-128 \
-DCMAKE_EXPORT_COMPILE_COMMANDS=ON \
-DCOMPILER_RT_BUILD_ORC=OFF \
-DCOMPILER_RT_BUILD_XRAY=OFF \
-DCOMPILER_RT_BUILD_MEMPROF=OFF \
-DCOMPILER_RT_BUILD_LIBFUZZER=OFF \
-DCOMPILER_RT_BUILD_SANITIZERS=OFF \
-DLLVM_CMAKE_DIR=$TRUNK_INSTALL_DIR \
-DLLVM_DIR=$TRUNK_INSTALL_DIR \
$TRUNK_ORIGIN_RPATH \
" 

if [ "$1" == "-h" ] || [ "$1" == "help" ] || [ "$1" == "-help" ] ; then 
  help_build_trunk
fi

if [ "$1" != "nocmake" ] && [ "$1" != "install" ] ; then
   echo 
   echo "This is a FRESH START. ERASING any previous builds in $BUILD_TRUNK/build/compiler-rt"
   echo "Use ""$0 nocmake"" or ""$0 install"" to avoid FRESH START."
   rm -rf $BUILD_TRUNK/build/compiler-rt
   mkdir -p $BUILD_TRUNK/build/compiler-rt
else
   if [ ! -d $BUILD_TRUNK/build/compiler-rt ] ; then 
      echo "ERROR: The build directory $BUILD_TRUNK/build/compiler-rt does not exist"
      echo "       run $0 without nocmake or install options. " 
      exit 1
   fi
fi

cd $BUILD_TRUNK/build/compiler-rt

if [ "$1" != "nocmake" ] && [ "$1" != "install" ] ; then
   echo
   echo " -----Running cmake ---- " 
   echo ${AOMP_CMAKE} $MYCMAKEOPTS $TRUNK_REPOS/llvm-project/compiler-rt
   ${AOMP_CMAKE} $MYCMAKEOPTS $TRUNK_REPOS/llvm-project/compiler-rt 2>&1
   if [ $? != 0 ] ; then 
      echo "ERROR cmake failed. Cmake flags"
      echo "      $MYCMAKEOPTS"
      exit 1
   fi
fi

echo
echo " -----Running $AOMP_NINJA_BIN ---- " 
echo $AOMP_NINJA_BIN -j $AOMP_JOB_THREADS
$AOMP_NINJA_BIN -j $AOMP_JOB_THREADS
if [ $? != 0 ] ; then
      echo "ERROR $AOMP_NINJA_BIN failed "
      exit 1
fi

if [ "$1" == "install" ] ; then
   echo " -----Installing to $TRUNK_INSTALL_DIR ---- "
   echo $AOMP_NINJA_BIN -j $AOMP_JOB_THREADS install
   $AOMP_NINJA_BIN -j $AOMP_JOB_THREADS install
   if [ $? != 0 ] ; then
      echo "ERROR $AOMP_NINJA_BIN install failed "
      exit 1
   fi
   echo
   echo "SUCCESSFUL INSTALL to $TRUNK_INSTALL_DIR "
   echo
else 
   echo 
   echo "SUCCESSFUL BUILD, please run:  $0 install"
   echo "  to install into $TRUNK_INSTALL_DIR"
   echo 
fi

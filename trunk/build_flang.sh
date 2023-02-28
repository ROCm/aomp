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

#  -DCMAKE_CXX_LINK_FLAGS='-Wl,-rpath,\$LD_LIBRARY_PATH' \

MYCMAKEOPTS="\
$_set_ninja_gan \
-DCMAKE_BUILD_TYPE=$BUILD_TYPE \
-DCMAKE_C_COMPILER=$TRUNK_INSTALL_DIR/bin/clang \
-DCMAKE_CXX_COMPILER=$TRUNK_INSTALL_DIR/bin/clang++ \
-DCMAKE_CXX_STANDARD=17 \
$AOMP_CCACHE_OPTS \
-DCMAKE_EXPORT_COMPILE_COMMANDS=ON \
-DFLANG_ENABLE_WERROR=ON \
-DLLVM_TARGETS_TO_BUILD=host \
-DLLVM_ENABLE_ASSERTIONS=ON \
-DLLVM_BUILD_MAIN_SRC_DIR=$TRUNK_REPOS/build/$LLVMPROJECT/lib/cmake/llvm \
-DLLVM_EXTERNAL_LIT=$TRUNK_REPOS/build/$LLVMPROJECT/bin/llvm-lit \
-DLLVM_LIT_ARGS=-v \
-DLLVM_DIR=$TRUNK_REPOS/build/$LLVMPROJECT/lib/cmake/llvm \
-DCLANG_DIR=$TRUNK_REPOS/build/$LLVMPROJECT/lib/cmake/clang \
-DMLIR_DIR=$TRUNK_REPOS/build/$LLVMPROJECT/lib/cmake/mlir \
-DCMAKE_INSTALL_PREFIX=$TRUNK_INSTALL_DIR \
"

if [ "$1" == "-h" ] || [ "$1" == "help" ] || [ "$1" == "-help" ] ; then 
  help_build_trunk
fi

if [ "$1" != "nocmake" ] && [ "$1" != "install" ] ; then
   echo 
   echo "This is a FRESH START. ERASING any previous builds in $BUILD_TRUNK/build/flang"
   echo "Use ""$0 nocmake"" or ""$0 install"" to avoid FRESH START."
   rm -rf $BUILD_TRUNK/build/flang
   mkdir -p $BUILD_TRUNK/build/flang
else
   if [ ! -d $BUILD_TRUNK/build/flang ] ; then 
      echo "ERROR: The build directory $BUILD_TRUNK/build/flang does not exist"
      echo "       run $0 without nocmake or install options. " 
      exit 1
   fi
fi

cd $BUILD_TRUNK/build/flang

if [ "$1" != "nocmake" ] && [ "$1" != "install" ] ; then
   echo
   echo " -----Running cmake ---- " 
   echo ${AOMP_CMAKE} $MYCMAKEOPTS $TRUNK_REPOS/$LLVMPROJECT/flang
   ${AOMP_CMAKE} $MYCMAKEOPTS $TRUNK_REPOS/$LLVMPROJECT/flang 2>&1
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

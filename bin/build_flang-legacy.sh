#!/bin/bash
# 
#  build_flang-legacy.sh:  Script to build the legacy flang binary driver
#         This driver will never call flang -fc1, it only calls binaries 
#             clang, flang1, flang2, build elsewhere
#
#
BUILD_TYPE=${BUILD_TYPE:-Release}

# --- Start standard header to set AOMP environment variables ----
realpath=`realpath $0`
thisdir=`dirname $realpath`
. $thisdir/aomp_common_vars
# --- end standard header ----

TARGETS_TO_BUILD="AMDGPU;${AOMP_NVPTX_TARGET}X86"

if [ $AOMP_STANDALONE_BUILD == 1 ] ; then
   standalone_word="_STANDALONE"
else
   standalone_word=""
fi

if [ "$AOMP_USE_NINJA" == 0 ] ; then
    AOMP_SET_NINJA_GEN=""
else
    AOMP_SET_NINJA_GEN="-G Ninja"
fi
# We need a version of ROCM llvm that supports legacy flang 
# via the link from flang to clang.  rocm 5.5 would be best. 
# This will enable removal of legacy flang driver support 
# from clang to make way for flang-new.  
export LLVM_DIR=/opt/rocm-5.4.3/llvm
MYCMAKEOPTS="\
-DLLVM_DIR=$LLVM_DIR \
-DCMAKE_BUILD_TYPE=$BUILD_TYPE \
-DCMAKE_C_COMPILER=$AOMP_INSTALL_DIR/bin/clang \
-DCMAKE_CXX_COMPILER=$AOMP_INSTALL_DIR/bin/clang++ \
-DCMAKE_CXX_STANDARD=17 \
-DBUILD_SHARED_LIBS=OFF \
-DCMAKE_INSTALL_PREFIX=$AOMP_INSTALL_DIR \
$AOMP_ORIGIN_RPATH \
$AOMP_SET_NINJA_GEN \
"

# Compiler-rt Sanitizer Build not available for legacy flang
if [ "$AOMP_BUILD_SANITIZER" == 'ON' ]; then
   echo "ERROR: legacy flang does not support ASAN"
   exit 1
fi

if [ "$1" == "-h" ] || [ "$1" == "help" ] || [ "$1" == "-help" ] ; then 
  help_build_aomp
fi

if [ $AOMP_STANDALONE_BUILD == 1 ] ; then 
   if [ ! -L $AOMP ] ; then 
     if [ -d $AOMP ] ; then 
        echo "ERROR: Directory $AOMP is a physical directory."
        echo "       It must be a symbolic link or not exist"
        exit 1
     fi
   fi
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
   echo "This is a FRESH START. ERASING any previous builds in $BUILD_DIR/build/flang-legacy"
   echo "Use ""$0 nocmake"" or ""$0 install"" to avoid FRESH START."
   rm -rf $BUILD_DIR/build/flang-legacy
   mkdir -p $BUILD_DIR/build/flang-legacy
else
   if [ ! -d $BUILD_DIR/build/flang-legacy ] ; then 
      echo "ERROR: The build directory $BUILD_DIR/build/flang-legacy does not exist"
      echo "       run $0 without nocmake or install options. " 
      exit 1
   fi
fi

cd $BUILD_DIR/build/flang-legacy 

if [ "$1" != "nocmake" ] && [ "$1" != "install" ] ; then
   echo
   echo " -----Running cmake ---- " 
   echo ${AOMP_CMAKE} $MYCMAKEOPTS  $AOMP_REPOS/$AOMP_FLANG_REPO_NAME/flang-legacy
   ${AOMP_CMAKE} $MYCMAKEOPTS  $AOMP_REPOS/$AOMP_FLANG_REPO_NAME/flang-legacy 2>&1
   if [ $? != 0 ] ; then 
      echo "ERROR cmake failed. Cmake flags"
      echo "      $MYCMAKEOPTS"
      exit 1
   fi
fi

echo
echo " ---  Running $AOMP_NINJA_BIN for $BUILD_DIR/build/flang-legacy ---- "
$AOMP_NINJA_BIN -j $AOMP_JOB_THREADS
if [ $? != 0 ] ; then
      echo " "
      echo "ERROR: $AOMP_NINJA_BIN -j $AOMP_JOB_THREADS  FAILED"
      echo "To restart:"
      echo "  cd $BUILD_DIR/build/flang-legacy"
      echo "  $AOMP_NINJA_BIN"
      exit 1
fi

if [ "$1" == "install" ] ; then
   echo " -----Installing to $AOMP_INSTALL_DIR ---- "
   $SUDO ${AOMP_CMAKE} --build . -j $AOMP_JOB_THREADS --target install
   if [ $? != 0 ] ; then
      echo "ERROR make install failed "
      exit 1
   fi
   echo
   echo "SUCCESSFUL INSTALL to $AOMP_INSTALL_DIR"
   echo
else 
   echo 
   echo "SUCCESSFUL BUILD, please run:  $0 install"
   echo "  to install into $AOMP"
   echo 
fi

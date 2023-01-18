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

DO_TESTS=${DO_TESTS:-"-DLLVM_BUILD_TESTS=ON -DLLVM_INCLUDE_TESTS=ON -DCLANG_INCLUDE_TESTS=ON"}
DO_TESTS=""

if [ "$AOMP_USE_NINJA" == 0 ] ; then
    _set_ninja_gan=""
else
    _set_ninja_gan="-G Ninja"
fi
if [ "$TRUNK_BUILD_CUDA" == 0 ] ; then
   _cuda_plugin="-DLIBOMPTARGET_BUILD_CUDA_PLUGIN=OFF"
   _targets_to_build="-DLLVM_TARGETS_TO_BUILD='X86;AMDGPU'"
else
   _cuda_plugin="-DLIBOMPTARGET_BUILD_CUDA_PLUGIN=ON"
   _targets_to_build="-DLLVM_TARGETS_TO_BUILD='X86;AMDGPU;NVPTX'"
fi

#  -DCMAKE_EXPORT_COMPILE_COMMANDS=ON \
#  -DFLANG_ENABLE_WERROR=ON \
#  -DLLVM_LIT_ARGS=-v \
#  -DLLVM_ENABLE_RUNTIMES='compiler-rt' \
#  -DLLVM_VERSION_SUFFIX=_$TRUNK_VERSION_STRING \
#  -DCLANG_VENDOR=AMD_TRUNK_$TRUNK_VERSION_STRING \
MYCMAKEOPTS="\
$_set_ninja_gan \
-DBUILD_SHARED_LIBS=ON \
-DLLVM_ENABLE_PROJECTS='lld;clang;mlir;openmp' \
-DCMAKE_BUILD_TYPE=$BUILD_TYPE \
-DCMAKE_INSTALL_PREFIX=$TRUNK_INSTALL_DIR \
$_targets_to_build \
-DLLVM_ENABLE_ASSERTIONS=ON \
-DLLVM_INSTALL_UTILS=ON \
-DCMAKE_CXX_STANDARD=17 \
$_cuda_plugin \
-DCLANG_DEFAULT_PIE_ON_LINUX=OFF \
" 

if [ "$1" == "-h" ] || [ "$1" == "help" ] || [ "$1" == "-help" ] ; then 
  help_build_trunk
fi

if [ ! -L $TRUNK_LINK ] ; then 
   if [ -d $TRUNK_LINK ] ; then 
     echo "ERROR: Directory $TRUNK_LINK is a physical directory."
     echo "       It must be a symbolic link or not exist"
     exit 1
   fi
fi

# Make sure we can update the install directory
if [ "$1" == "install" ] ; then 
   mkdir -p $TRUNK_INSTALL_DIR
   touch $TRUNK_INSTALL_DIR/testfile
   if [ $? != 0 ] ; then 
      echo "ERROR: No update access to $TRUNK_INSTALL_DIR"
      exit 1
   else
      echo "Successful No update access to $TRUNK_INSTALL_DIR"
   fi
   rm $TRUNK_INSTALL_DIR/testfile
fi

if [ "$1" != "nocmake" ] && [ "$1" != "install" ] ; then
   echo 
   echo "This is a FRESH START. ERASING any previous builds in $BUILD_TRUNK/build/llvm-project"
   echo "Use ""$0 nocmake"" or ""$0 install"" to avoid FRESH START."
   rm -rf $BUILD_TRUNK/build/llvm-project
   mkdir -p $BUILD_TRUNK/build/llvm-project
else
   if [ ! -d $BUILD_TRUNK/build/llvm-project ] ; then 
      echo "ERROR: The build directory $BUILD_TRUNK/build/llvm-project does not exist"
      echo "       run $0 without nocmake or install options. " 
      exit 1
   fi
fi

cd $BUILD_TRUNK/build/llvm-project

if [ "$1" != "nocmake" ] && [ "$1" != "install" ] ; then
   echo
   echo " -----Running cmake ---- " 
   echo ${AOMP_CMAKE} $MYCMAKEOPTS  $TRUNK_REPOS/llvm-project/llvm
   ${AOMP_CMAKE} $MYCMAKEOPTS  $TRUNK_REPOS/llvm-project/llvm 2>&1
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
   $AOMP_NINJA_BIN -j $AOMP_JOB_THREADS install
   if [ $? != 0 ] ; then
      echo "ERROR $AOMP_NINJA_BIN install failed "
      exit 1
   fi
   echo "latest" > $TRUNK_INSTALL_DIR/bin/versionrc
   echo " "
   echo "------ Linking $TRUNK_INSTALL_DIR to $TRUNK -------"
   if [ -L $TRUNK_LINK ] ; then 
      rm $TRUNK_LINK
   fi
   ln -sf $TRUNK_INSTALL_DIR $TRUNK_LINK
   echo
   echo "SUCCESSFUL INSTALL to $TRUNK_INSTALL_DIR with link to $TRUNK"
   echo
else 
   echo 
   echo "SUCCESSFUL BUILD, please run:  $0 install"
   echo "  to install into $TRUNK_INSTALL_DIR"
   echo 
fi

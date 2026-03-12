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

echo "LLVM PROJECTS TO BUILD:$TRUNK_PROJECTS_LIST"
DO_TESTS=${DO_TESTS:-"-DLLVM_BUILD_TESTS=ON -DLLVM_INCLUDE_TESTS=ON -DCLANG_INCLUDE_TESTS=ON"}

if [ "$AOMP_USE_NINJA" == 0 ] ; then
    _set_ninja_gan=""
else
    _set_ninja_gan="-G Ninja"
fi
if [ "$TRUNK_BUILD_CUDA" == 0 ] ; then
   _cuda_plugin="-DLIBOMPTARGET_BUILD_CUDA_PLUGIN=OFF"
   _targets_to_build="-DLLVM_TARGETS_TO_BUILD='X86;AMDGPU;SPIRV'"
   _plugins_to_build="-DLIBOMPTARGET_PLUGINS_TO_BUILD='amdgpu;host'"
   AOMP_NVPTX_CAPS_OPT=""
else
   _cuda_plugin="-DLIBOMPTARGET_BUILD_CUDA_PLUGIN=ON"
   _targets_to_build="-DLLVM_TARGETS_TO_BUILD='X86;AMDGPU;NVPTX;SPIRV'"
   _plugins_to_build="-DLIBOMPTARGET_PLUGINS_TO_BUILD='amdgpu;cuda;host'"
fi

#  -DCMAKE_EXPORT_COMPILE_COMMANDS=ON \
#  -DFLANG_ENABLE_WERROR=ON \
#  -DLLVM_VERSION_SUFFIX=_$TRUNK_VERSION_STRING \
#  -DCLANG_VENDOR=AMD_TRUNK_$TRUNK_VERSION_STRING \

# CMAKE options after LLVM_ENABLE_PROJECTS are NOT on build bot.
MYCMAKEOPTS="\
-DCMAKE_INSTALL_PREFIX=$TRUNK_INSTALL_DIR \
-DCMAKE_BUILD_TYPE=$BUILD_TYPE \
-DCLANG_DEFAULT_LINKER=lld \
$DO_TESTS \
$_targets_to_build \
-DLLVM_ENABLE_ASSERTIONS=ON \
-DLLVM_ENABLE_RUNTIMES='compiler-rt;flang-rt;libunwind;libcxx;libcxxabi;openmp;offload' \
$_plugins_to_build \
-DCOMPILER_RT_BUILD_ORC=OFF \
-DCOMPILER_RT_BUILD_XRAY=OFF \
-DCOMPILER_RT_BUILD_MEMPROF=OFF \
-DCOMPILER_RT_BUILD_LIBFUZZER=OFF \
-DCOMPILER_RT_BUILD_SANITIZERS=ON \
$AOMP_CCACHE_OPTS \
-DLLVM_ENABLE_PROJECTS='$TRUNK_PROJECTS_LIST' \
-DLLVM_INSTALL_UTILS=ON \
-DBUILD_SHARED_LIBS=ON \
-DCMAKE_CXX_STANDARD=17 \
$_cuda_plugin \
-DCLANG_DEFAULT_PIE_ON_LINUX=OFF \
-DFLANG_RUNTIME_F128_MATH_LIB=libquadmath \
$AOMP_GFXLIST_OPT \
$AOMP_NVPTX_CAPS_OPT \
$ENABLE_DEBUG_OPT \
$LLVM_FORCE_VC_REVISION_OPT \
$LLVM_FORCE_VC_REPOSITORY_OPT \
-DLLVM_ENABLE_PER_TARGET_RUNTIME_DIR=ON \
-DRUNTIMES_amdgcn-amd-amdhsa_LLVM_ENABLE_RUNTIMES='compiler-rt;libc;libcxx;libcxxabi;openmp;flang-rt' \
-DRUNTIMES_amdgcn-amd-amdhsa_LLVM_ENABLE_PER_TARGET_RUNTIME_DIR=ON \
-DRUNTIMES_amdgcn-amd-amdhsa_CACHE_FILES='$TRUNK_REPOS/$LLVMPROJECT/libcxx/cmake/caches/AMDGPU.cmake' \
-DRUNTIMES_amdgcn-amd-amdhsa_FLANG_RT_LIBC_PROVIDER=llvm \
-DRUNTIMES_amdgcn-amd-amdhsa_FLANG_RT_LIBCXX_PROVIDER=llvm \
"
if [ "$TRUNK_BUILD_CUDA" == 0 ] ; then
   MYCMAKEOPTS+="\
    -DLLVM_RUNTIME_TARGETS='default;amdgcn-amd-amdhsa' \
   "
else
   MYCMAKEOPTS+="\
    -DLLVM_RUNTIME_TARGETS='default;amdgcn-amd-amdhsa;nvptx64-nvidia-cuda' \
    -DRUNTIMES_nvptx64-nvidia-cuda_LLVM_ENABLE_RUNTIMES="compiler-rt;libc;openmp;libcxx;libcxxabi;flang-rt" \
    -DRUNTIMES_nvptx64-nvidia-cuda_LLVM_ENABLE_PER_TARGET_RUNTIME_DIR=ON \
    -DRUNTIMES_nvptx64-nvidia-cuda_CACHE_FILES='$TRUNK_REPOS/$LLVMPROJECT/libcxx/cmake/caches/NVPTX.cmake' \
    -DRUNTIMES_nvptx64-nvidia-cuda_FLANG_RT_LIBC_PROVIDER=llvm \
    -DRUNTIMES_nvptx64-nvidia-cuda_FLANG_RT_LIBCXX_PROVIDER=llvm \
   "
fi

# -DFLANG_RUNTIME_F128_MATH_LIB=libquadmath \

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
   echo "This is a FRESH START. ERASING any previous builds in $BUILD_TRUNK/build/$LLVMPROJECT"
   echo "Use ""$0 nocmake"" or ""$0 install"" to avoid FRESH START."
   rm -rf $BUILD_TRUNK/build/$LLVMPROJECT
   mkdir -p $BUILD_TRUNK/build/$LLVMPROJECT
else
   if [ ! -d $BUILD_TRUNK/build/$LLVMPROJECT ] ; then
      echo "ERROR: The build directory $BUILD_TRUNK/build/$LLVMPROJECT does not exist"
      echo "       run $0 without nocmake or install options. " 
      exit 1
   fi
fi

cd $BUILD_TRUNK/build/$LLVMPROJECT

if [ "$1" != "nocmake" ] && [ "$1" != "install" ] ; then
   echo
   echo " -----Running cmake ---- " 
   echo ${AOMP_CMAKE} $_set_ninja_gan $TRUNK_REPOS/$LLVMPROJECT/llvm -DCMAKE_BUILD_TYPE=$BUILD_TYPE -DLLVM_ENABLE_ASSERTIONS=ON '-DLLVM_LIT_ARGS=-vv --show-unsupported --show-xfail -j 16' $MYCMAKEOPTS
   ${AOMP_CMAKE} $_set_ninja_gan $TRUNK_REPOS/$LLVMPROJECT/llvm -DCMAKE_BUILD_TYPE=$BUILD_TYPE -DLLVM_ENABLE_ASSERTIONS=ON '-DLLVM_LIT_ARGS=-vv --show-unsupported --show-xfail -j 16' $MYCMAKEOPTS 2>&1
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
   # Create binary configs to avoid need to set LD_LIBRARY_PATH
   cat <<EOD > ${TRUNK_INSTALL_DIR}/bin/rpath.cfg
-Wl,-rpath=<CFGDIR>/../lib
-Wl,-rpath=<CFGDIR>/../lib/x86_64-unknown-linux-gnu
-L<CFGDIR>/../lib
-L<CFGDIR>/../lib/x86_64-unknown-linux-gnu
EOD
   ln -sf rpath.cfg ${TRUNK_INSTALL_DIR}/bin/clang++.cfg
   ln -sf rpath.cfg ${TRUNK_INSTALL_DIR}/bin/clang.cfg
   ln -sf rpath.cfg ${TRUNK_INSTALL_DIR}/bin/flang.cfg
   # flang-new also appears to be reading flang.cfg
   ln -sf rpath.cfg ${TRUNK_INSTALL_DIR}/bin/flang-new.cfg
   (
   # workaround for issue with triple subdir and shared builds
   # problem with libomptarget.so finding dependent libLLVM* libs
   cd ${TRUNK_INSTALL_DIR}/lib
   ln -sf x86_64-unknown-linux-gnu/*{.bc,.so,git} .
   )
   echo
   echo "SUCCESSFUL INSTALL to $TRUNK_INSTALL_DIR with link to $TRUNK"
   echo
else 
   echo 
   echo "SUCCESSFUL BUILD, please run:  $0 install"
   echo "  to install into $TRUNK_INSTALL_DIR"
   echo 
fi

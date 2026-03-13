#!/bin/bash
# 
#  build_project.sh:  Script to build the trunk compiler. 
#
BUILD_TYPE=${BUILD_TYPE:-Release}
# --- Start standard header to set AOMP environment variables ----
realpath=$(realpath "$0")
thisdir=$(dirname "$realpath")
. "$thisdir/trunk_common_vars"
# --- end standard header ----

declare -a _qmathopt
declare -a _amdflangrtopt
# Enable AMD-specific Fortran runtime extensions if not skipped
_amdflangrtopt=(-DFLANG_RT_INCLUDE_AMD=ON)
# Enable support for real(kind=16) via libquadmath
_qmathopt=(-DFLANG_RUNTIME_F128_MATH_LIB=libquadmath)

echo "LLVM PROJECTS TO BUILD:$TRUNK_PROJECTS_LIST"
#DO_TESTS=${DO_TESTS:-"-DLLVM_BUILD_TESTS=ON -DLLVM_INCLUDE_TESTS=ON -DCLANG_INCLUDE_TESTS=ON"}

if [ "$TRUNK_BUILD_CUDA" == 0 ] ; then
   _targets_to_build="-DLLVM_TARGETS_TO_BUILD='X86;AMDGPU;SPIRV'"
   _plugins_to_build="-DLIBOMPTARGET_PLUGINS_TO_BUILD='amdgpu;host'"
else
   _targets_to_build="-DLLVM_TARGETS_TO_BUILD='X86;AMDGPU;NVPTX;SPIRV'"
   _plugins_to_build="-DLIBOMPTARGET_PLUGINS_TO_BUILD='amdgpu;cuda;host'"
fi

if [ "$AOMP_USE_NINJA" == 0 ] ; then
    AOMP_SET_NINJA_GEN=()
else
    AOMP_SET_NINJA_GEN=(-G Ninja)
fi

MYCMAKEOPTS=(-DCMAKE_BUILD_TYPE="$BUILD_TYPE"
-DCMAKE_INSTALL_PREFIX="$TRUNK_INSTALL_DIR"
-DCLANG_DEFAULT_LINKER=lld
"${AOMP_SET_NINJA_GEN[@]}"
"${_qmathopt[@]}"
"${_amdflangrtopt[@]}"
"$_targets_to_build"
-DLLVM_ENABLE_ASSERTIONS=ON
"$_plugins_to_build"
"${AOMP_CCACHE_OPTS[@]}"
-DLLVM_INCLUDE_TESTS=Off
-DLLVM_INCLUDE_EXAMPLES=Off
-DCOMPILER_RT_BUILD_ORC=Off
-DCOMPILER_RT_BUILD_XRAY=Off
-DCOMPILER_RT_BUILD_MEMPROF=Off
-DCOMPILER_RT_BUILD_LIBFUZZER=Off
-DLLVM_ENABLE_PROJECTS="$TRUNK_PROJECTS_LIST"
-DLLVM_INSTALL_UTILS=ON
-DBUILD_SHARED_LIBS=ON
-DCMAKE_CXX_STANDARD=17
-DCLANG_DEFAULT_PIE_ON_LINUX=Off
-DFLANG_RUNTIME_F128_MATH_LIB=libquadmath
-DLLVM_ENABLE_RUNTIMES='libcxx;libcxxabi;libunwind;openmp;offload;compiler-rt;flang-rt'
-DLIBCXX_ENABLE_STATIC=ON
-DLIBCXXABI_ENABLE_STATIC=ON
-DLLVM_RUNTIME_TARGETS='default;amdgcn-amd-amdhsa;nvptx64-nvidia-cuda'
-DRUNTIMES_amdgcn-amd-amdhsa_LLVM_ENABLE_PER_TARGET_RUNTIME_DIR=ON
-DRUNTIMES_amdgcn-amd-amdhsa_LLVM_ENABLE_RUNTIMES='compiler-rt;libc;libcxx;libcxxabi;flang-rt;openmp'
-DRUNTIMES_amdgcn-amd-amdhsa_FLANG_RT_LIBC_PROVIDER=llvm
-DRUNTIMES_amdgcn-amd-amdhsa_FLANG_RT_LIBCXX_PROVIDER=llvm
-DRUNTIMES_amdgcn-amd-amdhsa_CACHE_FILES="'$TRUNK_REPOS/${LLVMPROJECT}/compiler-rt/cmake/caches/GPU.cmake;$TRUNK_REPOS/${LLVMPROJECT}/libcxx/cmake/caches/AMDGPU.cmake'"
-DRUNTIMES_nvptx64-nvidia-cuda_LLVM_ENABLE_PER_TARGET_RUNTIME_DIR=ON
-DRUNTIMES_nvptx64-nvidia-cuda_LLVM_ENABLE_RUNTIMES='compiler-rt;libc;libcxx;libcxxabi;flang-rt;openmp'
-DRUNTIMES_nvptx64-nvidia-cuda_FLANG_RT_LIBC_PROVIDER=llvm
-DRUNTIMES_nvptx64-nvidia-cuda_FLANG_RT_LIBCXX_PROVIDER=llvm
-DRUNTIMES_nvptx64-nvidia-cuda_CACHE_FILES="'$TRUNK_REPOS/${LLVMPROJECT}/compiler-rt/cmake/caches/GPU.cmake;$TRUNK_REPOS/${LLVMPROJECT}/libcxx/cmake/caches/NVPTX.cmake'")

if [ "$1" == "-h" ] || [ "$1" == "help" ] || [ "$1" == "-help" ] ; then 
  help_build_trunk
fi

if [ ! -L "$TRUNK_LINK" ] ; then 
   if [ -d "$TRUNK_LINK" ] ; then 
     echo "ERROR: Directory $TRUNK_LINK is a physical directory."
     echo "       It must be a symbolic link or not exist"
     exit 1
   fi
fi

# Make sure we can update the install directory
if [ "$1" == "install" ] ; then 
   mkdir -p "$TRUNK_INSTALL_DIR"
   if ! touch "$TRUNK_INSTALL_DIR/testfile" ; then
      echo "ERROR: No update access to $TRUNK_INSTALL_DIR"
      exit 1
   fi
   echo "Successful update access to $TRUNK_INSTALL_DIR"
   rm "$TRUNK_INSTALL_DIR"/testfile
fi

if [ "$1" != "nocmake" ] && [ "$1" != "install" ] ; then
   echo 
   echo "This is a FRESH START. ERASING any previous builds in $BUILD_TRUNK/build/$LLVMPROJECT"
   echo "Use ""$0 nocmake"" or ""$0 install"" to avoid FRESH START."
   rm -rf "$BUILD_TRUNK"/build/"$LLVMPROJECT"
   mkdir -p "$BUILD_TRUNK"/build/"$LLVMPROJECT"
else
   if [ ! -d "$BUILD_TRUNK/build/$LLVMPROJECT" ] ; then
      echo "ERROR: The build directory $BUILD_TRUNK/build/$LLVMPROJECT does not exist"
      echo "       run $0 without nocmake or install options. " 
      exit 1
   fi
fi

cd "$BUILD_TRUNK/build/$LLVMPROJECT" || exit 1

if [ "$1" != "nocmake" ] && [ "$1" != "install" ] ; then
   echo
   echo " -----Running cmake ---- "
   MYLITOPTS=(-DLLVM_LIT_ARGS='-vv --show-unsupported --show-xfail -j 16')
   echo "${AOMP_CMAKE}" "$(shquot "${MYLITOPTS[@]}")" \
                        "$(shquot "${MYCMAKEOPTS[@]}")" \
                        "$TRUNK_REPOS/${LLVMPROJECT}/llvm"

   if ! ${AOMP_CMAKE} "${MYLITOPTS[@]}" \
                      "${MYCMAKEOPTS[@]}" \
                      "$TRUNK_REPOS/${LLVMPROJECT}/llvm" 2>&1; then
      echo "ERROR cmake failed. Cmake flags"
      echo "      $(shquot "${MYCMAKEOPTS[@]}")"
      exit 1
   fi
fi

if [ "$1" = "cmake" ]; then
   exit 0
fi

if [ "$AOMP_LIMIT_FLANG" == "1" ] ; then
   # Required for building flang on memory limited systems.
   echo "${AOMP_CMAKE} --build . -- -j $AOMP_JOB_THREADS clang lld compiler-rt"
   ${AOMP_CMAKE} --build . -- -j "$AOMP_JOB_THREADS" clang lld compiler-rt

   echo "${AOMP_CMAKE} --build . -- -j $AOMP_FLANG_THREADS flang"
   ${AOMP_CMAKE} --build . -- -j "$AOMP_FLANG_THREADS" flang
fi

# Build llvm-project in one step
echo " -----Running CMAKE in ${PWD} ---- "
echo "${AOMP_CMAKE} --build . -j $AOMP_JOB_THREADS"
if ! ${AOMP_CMAKE} --build . -j "$AOMP_JOB_THREADS"; then
   echo "ERROR make -j $AOMP_JOB_THREADS failed"
   exit 1
fi

if [ "$1" == "install" ] ; then
   echo " -----Installing to $TRUNK_INSTALL_DIR ---- "
   echo "$AOMP_CMAKE --install ."
   if ! "$AOMP_CMAKE" --install . ; then
      echo "ERROR make install failed "
      exit 1
   fi

   echo "latest" > "$TRUNK_INSTALL_DIR"/bin/versionrc
   echo " "
   echo "------ Linking $TRUNK_INSTALL_DIR to $TRUNK -------"
   if [ -L "$TRUNK_LINK" ] ; then 
      rm "$TRUNK_LINK"
   fi
   ln -sf "$TRUNK_INSTALL_DIR" "$TRUNK_LINK"
   # Create binary configs to avoid need to set LD_LIBRARY_PATH
   cat <<EOD > "${TRUNK_INSTALL_DIR}"/bin/rpath.cfg
-Wl,-rpath=<CFGDIR>/../lib
-Wl,-rpath=<CFGDIR>/../lib/x86_64-unknown-linux-gnu
-L<CFGDIR>/../lib
-L<CFGDIR>/../lib/x86_64-unknown-linux-gnu
EOD
   ln -sf rpath.cfg "${TRUNK_INSTALL_DIR}"/bin/clang++.cfg
   ln -sf rpath.cfg "${TRUNK_INSTALL_DIR}"/bin/clang.cfg
   ln -sf rpath.cfg "${TRUNK_INSTALL_DIR}"/bin/flang.cfg
   # flang-new also appears to be reading flang.cfg
   ln -sf rpath.cfg "${TRUNK_INSTALL_DIR}"/bin/flang-new.cfg
   (
   # workaround for issue with triple subdir and shared builds
   # problem with libomptarget.so finding dependent libLLVM* libs
   cd "${TRUNK_INSTALL_DIR}"/lib || exit 1
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

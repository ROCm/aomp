#!/bin/bash
#
#  build_offload.sh:  Script to build the AOMP runtime libraries and debug libraries.  
#                This script will install in location defined by AOMP env variable
#
# --- Start standard header ----
function getdname(){
   local __DIRN=`dirname "$1"`
   if [ "$__DIRN" = "." ] ; then
      __DIRN=$PWD;
   else
      if [ ${__DIRN:0:1} != "/" ] ; then
         if [ ${__DIRN:0:2} == ".." ] ; then
               __DIRN=`dirname $PWD`/${__DIRN:3}
         else
            if [ ${__DIRN:0:1} = "." ] ; then
               __DIRN=$PWD/${__DIRN:2}
            else
               __DIRN=$PWD/$__DIRN
            fi
         fi
      fi
   fi
   echo $__DIRN
}
thisdir=$(getdname $0)
[ ! -L "$0" ] || thisdir=$(getdname `readlink "$0"`)
. $thisdir/aomp_common_vars
# --- end standard header ----

INSTALL_OPENMP=${INSTALL_OPENMP:-$AOMP_INSTALL_DIR}
if [ "$1" == "-h" ] || [ "$1" == "help" ] || [ "$1" == "-help" ] ; then 
  help_build_aomp
fi
AOMP_PROJECT_REPO_NAME="llvm-project"
REPO_BRANCH=$AOMP_PROJECT_REPO_BRANCH
REPO_DIR=$AOMP_REPOS/$AOMP_PROJECT_REPO_NAME
#checkrepo

if [ "$AOMP_BUILD_CUDA" == 1 ] ; then
   CUDAH=`find $CUDAT -type f -name "cuda.h" 2>/dev/null`
   if [ "$CUDAH" == "" ] ; then
      CUDAH=`find $CUDAINCLUDE -type f -name "cuda.h" 2>/dev/null`
   fi
   if [ "$CUDAH" == "" ] ; then
      echo
      echo "ERROR:  THE cuda.h FILE WAS NOT FOUND WITH ARCH $AOMP_PROC"
      echo "        A CUDA installation is necessary to build libomptarget deviceRTLs"
      echo "        Please install CUDA to build offload"
      echo
      exit 1
   fi
   # I don't see now nvcc is called, but this eliminates the deprecated warnings
   export CUDAFE_FLAGS="-w"
fi

#if [ ! -d $AOMP_REPOS/$AOMP_PROJECT_REPO_NAME ] ; then
#   echo "ERROR:  Missing repository $AOMP_REPOS/$AOMP_PROJECT_REPO_NAME "
#   echo "        Consider setting env variables AOMP_REPOS and/or AOMP_PROJECT_REPO_NAME "
#   exit 1
#fi

# Make sure we can update the install directory
if [ "$1" == "install" ] ; then 
   $SUDO mkdir -p $INSTALL_OPENMP
   $SUDO touch $INSTALL_OPENMP/testfile
   if [ $? != 0 ] ; then 
      echo "ERROR: No update access to $INSTALL_OPENMP"
      exit 1
   fi
   $SUDO rm $INSTALL_OPENMP/testfile
fi

if [ "$AOMP_BUILD_CUDA" == 1 ] ; then
   if [ -f $CUDABIN/nvcc ] ; then
      CUDAVER=`$CUDABIN/nvcc --version | grep compilation | cut -d" " -f5 | cut -d"." -f1 `
      echo "CUDA VERSION IS $CUDAVER"
   fi
fi

GFXSEMICOLONS=`echo $GFXLIST | tr ' ' ';' `
#COMMON_CMAKE_OPTS="#-DOPENMP_TEST_C_COMPILER=$AOMP/bin/clang
#-DOPENMP_TEST_CXX_COMPILER=$AOMP/bin/clang++

# FIXME: Remove CMAKE_CXX_FLAGS and CMAKE_C_FLAGS when AFAR uses 5.3 ROCr.
COMMON_CMAKE_OPTS="-DDEVICELIBS_ROOT=$DEVICELIBS_ROOT
-DOPENMP_ENABLE_LIBOMPTARGET=1
-DOPENMP_ENABLE_LIBOMPTARGET_HSA=1
-DLIBOMPTARGET_AMDGCN_GFXLIST=$GFXSEMICOLONS
-DLIBOMP_COPY_EXPORTS=OFF
-DAOMP_STANDALONE_BUILD=$AOMP_STANDALONE_BUILD
-DROCM_DIR=$ROCM_DIR
-DLIBOMPTARGET_ENABLE_DEBUG=ON
-DCMAKE_INSTALL_PREFIX=$INSTALL_OPENMP
-DLLVM_INSTALL_PREFIX=$INSTALL_PREFIX/llvm
-DLLVM_MAIN_INCLUDE_DIR=$LLVM_PROJECT_ROOT/llvm/include
-DLIBOMPTARGET_LLVM_INCLUDE_DIRS=$LLVM_PROJECT_ROOT/llvm/include
-DCMAKE_C_COMPILER=$INSTALL_PREFIX/llvm/bin/clang
-DCMAKE_CXX_COMPILER=$INSTALL_PREFIX/llvm/bin/clang++
-DOPENMP_TEST_C_COMPILER=$INSTALL_PREFIX/llvm/bin/clang
-DOPENMP_TEST_CXX_COMPILER=$INSTALL_PREFIX/llvm/bin/clang++"

if [ "$AOMP_STANDALONE_BUILD" == 0 ]; then
  COMMON_CMAKE_OPTS="$COMMON_CMAKE_OPTS  $OPENMP_EXTRAS_ORIGIN_RPATH"
fi

if [[ "$ROCM_DIR" =~ "/opt/rocm" ]]; then
  COMMON_CMAKE_OPTS="$COMMON_CMAKE_OPTS -DCMAKE_PREFIX_PATH=$OUT_DIR/build/devicelibs;$INSTALL_PREFIX;$ROCM_DIR;$ROCM_DIR/include/hsa;$OUT_DIR;$ROCM_CMAKECONFIG_PATH;$INSTALL_PREFIX/llvm/lib"
else
  COMMON_CMAKE_OPTS="$COMMON_CMAKE_OPTS -DCMAKE_PREFIX_PATH=$OUT_DIR/build/devicelibs;$ROCM_DIR;$ROCM_DIR/include/hsa;$ROCM_CMAKECONFIG_PATH;$INSTALL_PREFIX/llvm/lib"
fi

if [ "$AOMP_BUILD_CUDA" == 1 ] ; then
   COMMON_CMAKE_OPTS="$COMMON_CMAKE_OPTS
-DLIBOMPTARGET_NVPTX_ENABLE_BCLIB=ON
-DLIBOMPTARGET_NVPTX_CUDA_COMPILER=$AOMP/bin/clang++
-DLIBOMPTARGET_NVPTX_BC_LINKER=$AOMP/bin/llvm-link
-DLIBOMPTARGET_NVPTX_COMPUTE_CAPABILITIES=$NVPTXGPUS"
fi

if [ "$AOMP_BUILD_SANITIZER" == 1 ]; then
  ASAN_LIB_PATH=$($INSTALL_PREFIX/llvm/bin/clang --print-runtime-dir)
  if [ ! -d "$ASAN_LIB_PATH" ]; then
    CLANG_RES_DIR=$($INSTALL_PREFIX/llvm/bin/clang --print-resource-dir)
    ASAN_LIB_PATH="$CLANG_RES_DIR/lib/linux"
  fi
  ASAN_RPATH_FLAGS="-Wl,-rpath=$ASAN_LIB_PATH -L$ASAN_LIB_PATH"
  CXXFLAGS="$CXXFLAGS $ASAN_RPATH_FLAGS -I$ROCM_DIR/include -I$ROCM_DIR/include/hsa"
  CFLAGS="$CFLAGS $ASAN_RPATH_FLAGS -I$ROCM_DIR/include -I$ROCM_DIR/include/hsa"
  LDFLAGS=$LDFLAGS
fi

if [ "$1" != "nocmake" ] && [ "$1" != "install" ] ; then 

   echo " " 
   echo "This is a FRESH START. ERASING any previous builds in $BUILD_DIR/offload "
   echo "Use ""$0 nocmake"" or ""$0 install"" to avoid FRESH START."

   if [ $COPYSOURCE ] ; then
      mkdir -p $BUILD_DIR/$AOMP_PROJECT_REPO_NAME
      echo rsync -av --delete $AOMP_REPOS/$AOMP_PROJECT_REPO_NAME/offload  $BUILD_DIR/$AOMP_PROJECT_REPO_NAME
      rsync -av --delete $AOMP_REPOS/$AOMP_PROJECT_REPO_NAME/offload  $BUILD_DIR/$AOMP_PROJECT_REPO_NAME
   fi

      echo rm -rf $BUILD_DIR/build/offload
      rm -rf $BUILD_DIR/build/offload
      #MYCMAKEOPTS="$COMMON_CMAKE_OPTS -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_FLAGS=-g =DCMAKE_C_FLAGS=-g $AOMP_ORIGIN_RPATH -DROCM_DIR=$ROCM_DIR -DAOMP_STANDALONE_BUILD=$AOMP_STANDALONE_BUILD"
      MYCMAKEOPTS="$COMMON_CMAKE_OPTS \
      -DCMAKE_BUILD_TYPE=Release"
      if [ "$AOMP_BUILD_SANITIZER" != 1 ]; then
         mkdir -p $BUILD_DIR/build/offload
         cd $BUILD_DIR/build/offload
         echo " -----Running offload cmake ---- "
         if [ $COPYSOURCE ] ; then
            echo ${AOMP_CMAKE} $MYCMAKEOPTS -DCMAKE_C_FLAGS="-I$ROCM_DIR/include" -DCMAKE_CXX_FLAGS="-I$ROCM_DIR/include" $BUILD_DIR/$AOMP_PROJECT_REPO_NAME/offload
            env "$@" ${AOMP_CMAKE} $MYCMAKEOPTS -DCMAKE_C_FLAGS="-I$ROCM_DIR/include" -DCMAKE_CXX_FLAGS="-I$ROCM_DIR/include" $BUILD_DIR/$AOMP_PROJECT_REPO_NAME/offload
         else
            echo ${AOMP_CMAKE} $MYCMAKEOPTS -DCMAKE_C_FLAGS="-I$ROCM_DIR/include" -DCMAKE_CXX_FLAGS="-I$ROCM_DIR/include" $AOMP_REPOS/$AOMP_PROJECT_REPO_NAME/offload
            env "$@" ${AOMP_CMAKE} $MYCMAKEOPTS -DCMAKE_C_FLAGS="-I$ROCM_DIR/include" -DCMAKE_CXX_FLAGS="-I$ROCM_DIR/include" $AOMP_REPOS/../$AOMP_PROJECT_REPO_NAME/offload
         fi
         if [ $? != 0 ] ; then
            echo "ERROR offload cmake failed. Cmake flags"
            echo "      $MYCMAKEOPTS"
            exit 1
         fi
      fi

      if [ "$AOMP_BUILD_SANITIZER" == 1 ]; then
         ASAN_CMAKE_OPTS="$MYCMAKEOPTS -DLLVM_ENABLE_PER_TARGET_RUNTIME_DIR=OFF -DSANITIZER_AMDGPU=1 -DOFFLOAD_LIBDIR_SUFFIX=/asan"
         mkdir -p $BUILD_DIR/build/offload/asan
         cd $BUILD_DIR/build/offload/asan
         echo " ------Running offload-asan cmake ---- "
         if [ $COPYSOURCE ] ; then
            echo ${AOMP_CMAKE} $ASAN_CMAKE_OPTS -DCMAKE_C_FLAGS="'$CFLAGS'" -DCMAKE_CXX_FLAGS="'$CXXFLAGS'" $BUILD_DIR/$AOMP_PROJECT_REPO_NAME/offload
            env "$@" ${AOMP_CMAKE} $ASAN_CMAKE_OPTS -DCMAKE_C_FLAGS="'$CFLAGS'" -DCMAKE_CXX_FLAGS="'$CXXFLAGS'" $BUILD_DIR/$AOMP_PROJECT_REPO_NAME/offload
         else
            echo ${AOMP_CMAKE} $ASAN_CMAKE_OPTS -DCMAKE_C_FLAGS="'$CFLAGS'" -DCMAKE_CXX_FLAGS="'$CXXFLAGS'" $AOMP_REPOS/$AOMP_PROJECT_REPO_NAME/offload
            env "$@" ${AOMP_CMAKE} $ASAN_CMAKE_OPTS -DCMAKE_C_FLAGS="'$CFLAGS'" -DCMAKE_CXX_FLAGS="'$CXXFLAGS'" $AOMP_REPOS/../$AOMP_PROJECT_REPO_NAME/offload
         fi
         if [ $? != 0 ] ; then
            echo "ERROR offload-asan cmake failed. Cmake flags"
            echo "      $ASAN_CMAKE_OPTS"
            exit 1
         fi
      fi

      echo rm -rf $BUILD_DIR/build/offload_debug
      rm -rf $BUILD_DIR/build/offload_debug
      #MYCMAKEOPTS="$COMMON_CMAKE_OPTS -DLIBOMPTARGET_NVPTX_DEBUG=ON -DCMAKE_BUILD_TYPE=Debug $AOMP_ORIGIN_RPATH -DROCM_DIR=$ROCM_DIR -DAOMP_STANDALONE_BUILD=$AOMP_STANDALONE_BUILD"
      MYCMAKEOPTS="$COMMON_CMAKE_OPTS \
      -DCMAKE_BUILD_TYPE=Debug \
      -DLIBOMPTARGET_NVPTX_DEBUG=ON \
      -DLIBOMP_ARCH=x86_64 \
      -DLIBOMP_OMPT_SUPPORT=ON \
      -DLIBOMP_USE_DEBUGGER=ON \
      -DLIBOMP_CPPFLAGS='-O0' \
      -DLIBOMP_OMPD_SUPPORT=ON \
      -DLIBOMP_OMPT_DEBUG=ON \
      -DOPENMP_SOURCE_DEBUG_MAP="\""-fdebug-prefix-map=$AOMP_REPOS/$AOMP_PROJECT_REPO_NAME/offload=$ROCM_INSTALL_PATH/llvm/lib-debug/src/offload"\"""

      # Only use CMAKE_CXX/C_FLAGS on non-asan builds as these will overwrite the asan flags
      if [ "$AOMP_BUILD_SANITIZER" == 1  ]; then
         ASAN_CMAKE_OPTS="$MYCMAKEOPTS -DLLVM_ENABLE_PER_TARGET_RUNTIME_DIR=OFF -DSANITIZER_AMDGPU=1 -DOFFLOAD_LIBDIR_SUFFIX=-debug/asan"
      fi

      MYCMAKEOPTS="$MYCMAKEOPTS -DOFFLOAD_LIBDIR_SUFFIX=-debug"
      if [ "$AOMP_BUILD_SANITIZER" != 1 ]; then
         mkdir -p $BUILD_DIR/build/offload_debug
         cd $BUILD_DIR/build/offload_debug
         echo
         echo " -----Running offload cmake for debug ---- "
         if [ $COPYSOURCE ] ; then
            echo ${AOMP_CMAKE} $MYCMAKEOPTS -DCMAKE_C_FLAGS="-g -I$ROCM_DIR/include" -DCMAKE_CXX_FLAGS="-g -I$ROCM_DIR/include" $BUILD_DIR/$AOMP_PROJECT_REPO_NAME/offload
            env "$@" ${AOMP_CMAKE} $MYCMAKEOPTS -DCMAKE_C_FLAGS="-g -I$ROCM_DIR/include" -DCMAKE_CXX_FLAGS="-g -I$ROCM_DIR/include" $BUILD_DIR/$AOMP_PROJECT_REPO_NAME/offload
         else
            # FIXME: Remove CMAKE_CXX_FLAGS and CMAKE_C_FLAGS when AFAR uses 5.3 ROCr.
            echo ${AOMP_CMAKE} $MYCMAKEOPTS -DCMAKE_C_FLAGS="-g -I$ROCM_DIR/include" -DCMAKE_CXX_FLAGS="-g -I$ROCM_DIR/include" $AOMP_REPOS/../$AOMP_PROJECT_REPO_NAME/offload
            env "$@" ${AOMP_CMAKE} $MYCMAKEOPTS -DCMAKE_C_FLAGS="-g -I$ROCM_DIR/include" -DCMAKE_CXX_FLAGS="-g -I$ROCM_DIR/include" $AOMP_REPOS/../$AOMP_PROJECT_REPO_NAME/offload
         fi
         if [ $? != 0 ] ; then
            echo "ERROR offload debug cmake failed. Cmake flags"
            echo "      $MYCMAKEOPTS"
            exit 1
         fi
      fi

      if [ "$AOMP_BUILD_SANITIZER" == 1 ]; then
         mkdir -p $BUILD_DIR/build/offload_debug/asan
         cd $BUILD_DIR/build/offload_debug/asan
         echo
         echo " -----Running offload cmake for debug-asan ---- "
         if [ $COPYSOURCE ] ; then
            echo ${AOMP_CMAKE} $ASAN_CMAKE_OPTS -DCMAKE_C_FLAGS="'$CFLAGS -I$ROCM_DIR/include'" -DCMAKE_CXX_FLAGS="'$CXXFLAGS -I$ROCM_DIR/include'" $BUILD_DIR/$AOMP_PROJECT_REPO_NAME/offload
            env "$@" ${AOMP_CMAKE} $ASAN_CMAKE_OPTS -DCMAKE_C_FLAGS="'$CFLAGS -I$ROCM_DIR/include'" -DCMAKE_CXX_FLAGS="'$CXXFLAGS -I$ROCM_DIR/include'" $BUILD_DIR/$AOMP_PROJECT_REPO_NAME/offload
         else
            # FIXME: Remove CMAKE_CXX_FLAGS and CMAKE_C_FLAGS when AFAR uses 5.3 ROCr.
            echo ${AOMP_CMAKE} $ASAN_CMAKE_OPTS -DCMAKE_C_FLAGS="'$CFLAGS -I$ROCM_DIR/include'" -DCMAKE_CXX_FLAGS="'$CXXFLAGS -I$ROCM_DIR/include'" $AOMP_REPOS/../$AOMP_PROJECT_REPO_NAME/offload
            env "$@" ${AOMP_CMAKE} $ASAN_CMAKE_OPTS -DCMAKE_C_FLAGS="'$CFLAGS -I$ROCM_DIR/include'" -DCMAKE_CXX_FLAGS="'$CXXFLAGS -I$ROCM_DIR/include'" $AOMP_REPOS/../$AOMP_PROJECT_REPO_NAME/offload
         fi
         if [ $? != 0 ] ; then
            echo "ERROR offload debug-asan cmake failed. Cmake flags"
            echo "      $ASAN_CMAKE_OPTS"
            exit 1
         fi
      fi
fi

if [ "$AOMP_BUILD_SANITIZER" != 1 ]; then
   cd $BUILD_DIR/build/offload
   echo
   echo " -----Running make for $BUILD_DIR/build/offload ---- "
   make -j $NUM_THREADS
   if [ $? != 0 ] ; then
      echo " "
      echo "ERROR: make -j $NUM_THREADS  FAILED"
      echo "To restart:"
      echo "  cd $BUILD_DIR/build/offload"
      echo "  make"
      exit 1
   fi
fi

if [ "$AOMP_BUILD_SANITIZER" == 1 ]; then
   cd $BUILD_DIR/build/offload/asan
   echo
   echo " ------ Running make for $BUILD_DIR/build/offload/asan ---- "
   make -j $NUM_THREADS
   if [ $? != 0 ] ; then
      echo " "
      echo "ERROR: make -j $NUM_THREADS  FAILED"
      echo "To restart:"
      echo "  cd $BUILD_DIR/build/offload/asan"
      echo "  make"
      exit 1
   fi
fi

if [ "$AOMP_BUILD_SANITIZER" != 1 ]; then
   cd $BUILD_DIR/build/offload_debug
   echo
   echo
   echo " -----Running make for $BUILD_DIR/build/offload_debug ---- "
   make -j $NUM_THREADS
   if [ $? != 0 ] ; then
      echo "ERROR make -j $NUM_THREADS failed"
      exit 1
   else
      if [ "$1" != "install" ] ; then
         echo
         echo "Successful build of ./build_offload.sh .  Please run:"
         echo "  ./build_offload.sh install "
         echo "to install into directory $INSTALL_OPENMP/lib and $INSTALL_OPENMP/lib-debug"
         echo
      fi
   fi
fi

if [ "$AOMP_BUILD_SANITIZER" == 1 ]; then
   cd $BUILD_DIR/build/offload_debug/asan
   echo
   echo
   echo " -----Running make for $BUILD_DIR/build/offload_debug/asan ---- "
   make -j $NUM_THREADS
   if [ $? != 0 ] ; then
      echo "ERROR make -j $NUM_THREADS failed"
      exit 1
   else
      if [ "$1" != "install" ] ; then
         echo
         echo "Successful ASan build of ./build_offload.sh .  Please run:"
         echo "  ./build_offload.sh install "
         echo "to install into directory $INSTALL_OPENMP/lib/asan and $INSTALL_OPENMP/lib-debug/asan"
         echo
      fi
   fi
fi

#  ----------- Install only if asked  ----------------------------
if [ "$1" == "install" ] ; then
      if [ "$AOMP_BUILD_SANITIZER" != 1 ]; then
         cd $BUILD_DIR/build/offload
         echo
         echo " -----Installing to $INSTALL_OPENMP/lib ----- "
         $SUDO make -j $NUM_THREADS install
         if [ $? != 0 ] ; then
            echo "ERROR make install failed "
            exit 1
         fi
      fi

      if [ "$AOMP_BUILD_SANITIZER" == 1 ]; then
         cd $BUILD_DIR/build/offload/asan
         echo
         echo " ----- Installing to $INSTALL_OPENMP/lib/asan ------ "
         $SUDO make -j $NUM_THREADS install
         if [ $? != 0 ]; then
            echo "ERROR make install failed for offload/asan"
            exit 1
         fi
      fi

      if [ "$AOMP_BUILD_SANITIZER" != 1 ]; then
         cd $BUILD_DIR/build/offload_debug
         echo
         echo " -----Installing to $INSTALL_OPENMP/lib-debug ---- "
         $SUDO make -j $NUM_THREADS install
         if [ $? != 0 ] ; then
            echo "ERROR make install failed "
            exit 1
         fi
      fi

      if [ "$AOMP_BUILD_SANITIZER" == 1 ]; then
         cd $BUILD_DIR/build/offload_debug/asan
         echo
         echo " ----- Installing to $INSTALL_OPENMP/lib-debug/asan ------ "
         $SUDO make -j $NUM_THREADS install
         if [ $? != 0 ]; then
            echo "ERROR make install failed for offload_debug/asan"
            exit 1
         fi
      fi
      # Copy selected debugable runtime sources into the installation lib-debug/src directory
      # to satisfy the above -fdebug-prefix-map.
      $SUDO mkdir -p $AOMP_INSTALL_DIR/lib-debug/src/offload/src
      echo cp -rp $LLVM_PROJECT_ROOT/offload/src $AOMP_INSTALL_DIR/lib-debug/src/offload
      $SUDO cp -rp $LLVM_PROJECT_ROOT/offload/src $AOMP_INSTALL_DIR/lib-debug/src/offload
      echo cp -rp $LLVM_PROJECT_ROOT/offload/plugins-nextgen $AOMP_INSTALL_DIR/lib-debug/src/offload
      $SUDO cp -rp $LLVM_PROJECT_ROOT/offload/plugins-nextgen $AOMP_INSTALL_DIR/lib-debug/src/offload
fi

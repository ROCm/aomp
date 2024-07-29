#!/bin/bash
#
#  build_offload.sh:  Script to build the AOMP runtime libraries and debug libraries.
#                This script will install in location defined by AOMP env variable
#

# --- Start standard header to set AOMP environment variables ----
realpath=`realpath $0`
thisdir=`dirname $realpath`
. $thisdir/aomp_common_vars
# --- end standard header ----

INSTALL_OFFLOAD=$AOMP_INSTALL_DIR/lib/llvm

if [ "$1" == "-h" ] || [ "$1" == "help" ] || [ "$1" == "-help" ] ; then
  help_build_aomp
fi

REPO_DIR=$AOMP_REPOS/$AOMP_PROJECT_REPO_NAME

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

if [ ! -d $AOMP_REPOS/$AOMP_PROJECT_REPO_NAME ] ; then
   echo "ERROR:  Missing repository $AOMP_REPOS/$AOMP_PROJECT_REPO_NAME "
   echo "        Consider setting env variables AOMP_REPOS and/or AOMP_PROJECT_REPO_NAME "
   exit 1
fi

# Make sure we can update the install directory
if [ "$1" == "install" ] ; then
   $SUDO mkdir -p $INSTALL_OFFLOAD
   $SUDO touch $INSTALL_OFFLOAD/testfile
   if [ $? != 0 ] ; then
      echo "ERROR: No update access to $INSTALL_OFFLOAD"
      exit 1
   fi
   $SUDO rm $INSTALL_OFFLOAD/testfile
fi

if [ "$AOMP_BUILD_CUDA" == 1 ] ; then
   if [ -f $CUDABIN/nvcc ] ; then
      CUDAVER=`$CUDABIN/nvcc --version | grep compilation | cut -d" " -f5 | cut -d"." -f1 `
      echo "CUDA VERSION IS $CUDAVER"
   fi
fi

if [ "$AOMP_USE_NINJA" == 0 ] ; then
    AOMP_SET_NINJA_GEN=""
else
    AOMP_SET_NINJA_GEN="-G Ninja"
fi

export LLVM_DIR=$AOMP_INSTALL_DIR
GFXSEMICOLONS=`echo $GFXLIST | tr ' ' ';' `
ALTAOMP=${ALTAOMP:-$AOMP}
COMMON_CMAKE_OPTS="$AOMP_SET_NINJA_GEN -DOPENMP_ENABLE_LIBOMPTARGET=1 \
-DCMAKE_INSTALL_PREFIX=$INSTALL_OFFLOAD \
-DOPENMP_TEST_C_COMPILER=$AOMP/lib/llvm/bin/clang \
-DOPENMP_TEST_CXX_COMPILER=$AOMP/lib/llvm/bin/clang++ \
-DCMAKE_C_COMPILER=$ALTAOMP/lib/llvm/bin/clang \
-DCMAKE_CXX_COMPILER=$ALTAOMP/lib/llvm/bin/clang++ \
-DLIBOMPTARGET_AMDGCN_GFXLIST=$GFXSEMICOLONS \
-DLIBOMPTARGET_ENABLE_DEBUG=ON \
-DOPENMP_LLVM_TOOLS_DIR=$AOMP_INSTALL_DIR/lib/llvm/bin \
-DLLVM_DIR=$LLVM_DIR"

if [ "$AOMP_STANDALONE_BUILD" == 0 ]; then
  COMMON_CMAKE_OPTS="$COMMON_CMAKE_OPTS
  -DLLVM_MAIN_INCLUDE_DIR=$LLVM_PROJECT_ROOT/llvm/include
  -DLIBOMPTARGET_LLVM_INCLUDE_DIRS=$LLVM_PROJECT_ROOT/llvm/include"
else
  COMMON_CMAKE_OPTS="$COMMON_CMAKE_OPTS \
  -DLLVM_MAIN_INCLUDE_DIR=$AOMP_REPOS/$AOMP_PROJECT_REPO_NAME/llvm/include \
  -DLIBOMPTARGET_LLVM_INCLUDE_DIRS=$AOMP_REPOS/$AOMP_PROJECT_REPO_NAME/llvm/include"
fi

if [ "$AOMP_BUILD_CUDA" == 1 ] ; then
   COMMON_CMAKE_OPTS="$COMMON_CMAKE_OPTS
-DLIBOMPTARGET_NVPTX_ENABLE_BCLIB=ON
-DLIBOMPTARGET_NVPTX_CUDA_COMPILER=$AOMP/lib/llvm/bin/clang++
-DLIBOMPTARGET_NVPTX_BC_LINKER=$AOMP/lib/llvm/bin/llvm-link
-DLIBOMPTARGET_NVPTX_COMPUTE_CAPABILITIES=$NVPTXGPUS"
else
#  Need to force CUDA off this way in case cuda is installed in this system
   COMMON_CMAKE_OPTS="$COMMON_CMAKE_OPTS
-DCUDA_TOOLKIT_ROOT_DIR=OFF"
fi

if [ "$AOMP_BUILD_SANITIZER" == 1 ]; then
   LDFLAGS="-fuse-ld=lld $ASAN_FLAGS"
fi

# This is how we tell the hsa plugin where to find hsa
export HSA_RUNTIME_PATH=$ROCM_DIR

#breaks build as it cant find rocm-path
#export HIP_DEVICE_LIB_PATH=$ROCM_DIR/lib

if [ "$1" != "nocmake" ] && [ "$1" != "install" ] ; then
   echo " "
   echo "This is a FRESH START. ERASING any previous builds in $BUILD_DIR/offload."
   echo "Use ""$0 nocmake"" or ""$0 install"" to avoid FRESH START."
   echo rm -rf $BUILD_DIR/build/offload
   rm -rf $BUILD_DIR/build/offload
   if [ "$AOMP_STANDALONE_BUILD" == 1 ]; then
     MYCMAKEOPTS="$COMMON_CMAKE_OPTS -DCMAKE_PREFIX_PATH=$AOMP_INSTALL_DIR/lib/llvm/lib/cmake -DCMAKE_BUILD_TYPE=Release $AOMP_ORIGIN_RPATH"
   else
     MYCMAKEOPTS="$COMMON_CMAKE_OPTS -DCMAKE_PREFIX_PATH=$INSTALL_PREFIX/lib/cmake -DCMAKE_BUILD_TYPE=Release $OPENMP_EXTRAS_ORIGIN_RPATH"
   fi
   if [ "$AOMP_LEGACY_OPENMP" == "1" ] && [ "$SANITIZER" != 1 ]; then
      echo " -----Running offload cmake ---- "
      mkdir -p $BUILD_DIR/build/offload
      cd $BUILD_DIR/build/offload
      echo ${AOMP_CMAKE} $MYCMAKEOPTS  $AOMP_REPOS/$AOMP_PROJECT_REPO_NAME/offload
      ${AOMP_CMAKE} $MYCMAKEOPTS $AOMP_REPOS/$AOMP_PROJECT_REPO_NAME/offload
      if [ $? != 0 ] ; then
         echo "ERROR offload cmake failed. Cmake flags"
         echo "      $MYCMAKEOPTS"
         exit 1
      fi
   fi
      if [ "$AOMP_BUILD_SANITIZER" == 1 ]; then
        if [ "$AOMP_STANDALONE_BUILD" == 1 ]; then
          ASAN_CMAKE_OPTS="$COMMON_CMAKE_OPTS -DCMAKE_PREFIX_PATH=$AOMP_INSTALL_DIR/lib/llvm/lib/asan/cmake;$AOMP_INSTALL_DIR/lib/llvm/lib/cmake  -DSANITIZER_AMDGPU=1 -DCMAKE_BUILD_TYPE=Release -DLLVM_ENABLE_PER_TARGET_RUNTIME_DIR=OFF $AOMP_ASAN_ORIGIN_RPATH"
        else
          ASAN_CMAKE_OPTS="$COMMON_CMAKE_OPTS -DCMAKE_PREFIX_PATH=$ROCM_CMAKECONFIG_PATH;$INSTALL_PREFIX/lib/llvm/lib/asan -DSANITIZER_AMDGPU=1 -DCMAKE_BUILD_TYPE=Release -DLLVM_ENABLE_PER_TARGET_RUNTIME_DIR=OFF $OPENMP_EXTRAS_ORIGIN_RPATH"
        fi
        echo " -----Running offload cmake for asan ---- "
        mkdir -p $BUILD_DIR/build/offload/asan
        cd $BUILD_DIR/build/offload/asan
        echo ${AOMP_CMAKE} $ASAN_CMAKE_OPTS -DCMAKE_C_FLAGS="'$ASAN_FLAGS'" -DCMAKE_CXX_FLAGS="'$ASAN_FLAGS'" -DOFFLOAD_LIBDIR_SUFFIX="/asan" $AOMP_REPOS/$AOMP_PROJECT_REPO_NAME/offload
        ${AOMP_CMAKE} $ASAN_CMAKE_OPTS -DCMAKE_C_FLAGS="'$ASAN_FLAGS'" -DCMAKE_CXX_FLAGS="'$ASAN_FLAGS'" -DOFFLOAD_LIBDIR_SUFFIX="/asan" $AOMP_REPOS/$AOMP_PROJECT_REPO_NAME/offload
        if [ $? != 0 ] ; then
           echo "ERROR offload cmake failed. Cmake flags"
           echo "      $ASAN_CMAKE_OPTS"
           exit 1
        fi
      fi

  # Build a dedicatd "performance" version of libomptarget
  if [ "$AOMP_BUILD_PERF" == "1" ]; then
    echo rm -rf $BUILD_DIR/build/offload_perf
    rm -rf $BUILD_DIR/build/offload_perf
    MYCMAKEOPTS="$COMMON_CMAKE_OPTS -DCMAKE_PREFIX_PATH=$AOMP_INSTALL_DIR/lib/llvm/lib-perf/cmake;$AOMP_INSTALL_DIR/lib/llvm/lib/cmake -DLIBOMPTARGET_ENABLE_DEBUG=OFF -DCMAKE_BUILD_TYPE=Release -DLIBOMPTARGET_PERF=ON -DOFFLOAD_LIBDIR_SUFFIX=-perf"
    mkdir -p $BUILD_DIR/build/offload_perf
    cd $BUILD_DIR/build/offload_perf
    echo " -----Running offload cmake for perf ---- "
    echo ${AOMP_CMAKE} $MYCMAKEOPTS  $AOMP_REPOS/$AOMP_PROJECT_REPO_NAME/offload $AOMP_ORIGIN_RPATH
    ${AOMP_CMAKE} $MYCMAKEOPTS  $AOMP_REPOS/$AOMP_PROJECT_REPO_NAME/offload $AOMP_ORIGIN_RPATH
    if [ $? != 0 ] ; then
       echo "error offload cmake failed. cmake flags"
       echo "      $MYCMAKEOPTS"
       exit 1
    fi
    if [ "$AOMP_BUILD_SANITIZER" == 1 ]; then
       ASAN_CMAKE_OPTS="$COMMON_CMAKE_OPTS -DCMAKE_PREFIX_PATH=$AOMP_INSTALL_DIR/lib/llvm/lib/asan/cmake;$AOMP_INSTALL_DIR/lib/llvm/lib/cmake  -DLIBOMPTARGET_ENABLE_DEBUG=OFF -DCMAKE_BUILD_TYPE=Release -DLLVM_ENABLE_PER_TARGET_RUNTIME_DIR=OFF -DLIBOMPTARGET_PERF=ON -DSANITIZER_AMDGPU=1 $AOMP_ASAN_ORIGIN_RPATH"
       echo " -----Running offload cmake for perf-asan ---- "
       mkdir -p $BUILD_DIR/build/offload_perf/asan
       cd $BUILD_DIR/build/offload_perf/asan
       echo ${AOMP_CMAKE} $ASAN_CMAKE_OPTS -DCMAKE_C_FLAGS="'$ASAN_FLAGS'" -DCMAKE_CXX_FLAGS="'$ASAN_FLAGS'" -DOFFLOAD_LIBDIR_SUFFIX="-perf/asan" $AOMP_REPOS/$AOMP_PROJECT_REPO_NAME/offload
       ${AOMP_CMAKE} $ASAN_CMAKE_OPTS -DCMAKE_C_FLAGS="'$ASAN_FLAGS'" -DCMAKE_CXX_FLAGS="'$ASAN_FLAGS'" -DOFFLOAD_LIBDIR_SUFFIX="-perf/asan" $AOMP_REPOS/$AOMP_PROJECT_REPO_NAME/offload
       if [ $? != 0 ] ; then
          echo "error offload cmake failed. cmake flags"
          echo "      $ASAN_CMAKE_OPTS"
          exit 1
       fi
    fi
  fi

   if [ "$AOMP_BUILD_DEBUG" == "1" ] ; then
      _ompd_dir="$AOMP_INSTALL_DIR/lib-debug/ompd"
      #  This is the new locationof the ompd directory
      [[ ! -d $_ompd_dir ]] && _ompd_dir="$AOMP_INSTALL_DIR/share/gdb/python/ompd"

      DEBUGCMAKEOPTS="
-DLIBOMPTARGET_NVPTX_DEBUG=ON \
-DLLVM_ENABLE_ASSERTIONS=ON \
-DCMAKE_BUILD_TYPE=Debug \
-DROCM_DIR=$ROCM_DIR \
-DLIBOMP_ARCH=x86_64 \
-DLIBOMP_OMPT_SUPPORT=ON \
-DLIBOMP_USE_DEBUGGER=ON \
-DLIBOMP_CPPFLAGS='-O0' \
-DLIBOMP_OMPD_SUPPORT=ON \
-DLIBOMP_OMPT_DEBUG=ON \
-DOPENMP_SOURCE_DEBUG_MAP="\""-fdebug-prefix-map=$AOMP_REPOS/$AOMP_PROJECT_REPO_NAME/offload=$_ompd_dir/src/offload"\"" "

      # The 'pip install --system' command is not supported on non-debian systems. This will disable
      # the system option if the debian_version file is not present.
      if [ ! -f /etc/debian_version ]; then
         echo "==> Non-Debian OS, disabling use of pip install --system"
         DEBUGCMAKEOPTS="$DEBUGCMAKEOPTS -DDISABLE_SYSTEM_NON_DEBIAN=1"
      fi

      # Redhat 7.6 does not have python36-devel package, which is needed for ompd compilation.
      # This is acquired through RH Software Collections.
      if [ -f /opt/rh/rh-python36/enable ]; then
         echo "==> Using python3.6 out of rh tools."
         DEBUGCMAKEOPTS="$DEBUGCMAKEOPTS -DPython3_ROOT_DIR=/opt/rh/rh-python36/root/bin -DPYTHON_HEADERS=/opt/rh/rh-python36/root/usr/include/python3.6m"
      fi

      echo
      if [ "$SANITIZER" != 1 ] ; then
         echo rm -rf $BUILD_DIR/build/offload_debug
         rm -rf $BUILD_DIR/build/offload_debug
         echo " -----Running offload cmake for debug ---- "
         mkdir -p $BUILD_DIR/build/offload_debug
         cd $BUILD_DIR/build/offload_debug
         if [ "$AOMP_STANDALONE_BUILD" == 1 ]; then
           PREFIX_PATH="-DCMAKE_PREFIX_PATH=$AOMP_INSTALL_DIR/lib/llvm/lib/cmake"
           MYCMAKEOPTS="$COMMON_CMAKE_OPTS $DEBUGCMAKEOPTS $AOMP_ORIGIN_RPATH"
         else
           PREFIX_PATH="-DCMAKE_PREFIX_PATH=$INSTALL_PREFIX/lib/cmake"
           MYCMAKEOPTS="$COMMON_CMAKE_OPTS $DEBUGCMAKEOPTS $OPENMP_EXTRAS_ORIGIN_RPATH"
         fi
         echo ${AOMP_CMAKE} $MYCMAKEOPTS $PREFIX_PATH -DCMAKE_C_FLAGS="$CFLAGS -g" -DCMAKE_CXX_FLAGS="$CXXFLAGS -g" -DOFFLOAD_LIBDIR_SUFFIX=-debug $AOMP_REPOS/$AOMP_PROJECT_REPO_NAME/offload
         ${AOMP_CMAKE} $MYCMAKEOPTS $PREFIX_PATH -DCMAKE_C_FLAGS="$CFLAGS -g" -DCMAKE_CXX_FLAGS="$CXXFLAGS -g" -DOFFLOAD_LIBDIR_SUFFIX=-debug $AOMP_REPOS/$AOMP_PROJECT_REPO_NAME/offload

         if [ $? != 0 ] ; then
            echo "ERROR offload debug cmake failed. Cmake flags"
            echo "      $MYCMAKEOPTS"
            exit 1
         fi
      fi
      if [ "$AOMP_BUILD_SANITIZER" == 1 ]; then
         ASAN_CMAKE_OPTS="$COMMON_CMAKE_OPTS $DEBUGCMAKEOPTS -DLLVM_ENABLE_PER_TARGET_RUNTIME_DIR=OFF -DSANITIZER_AMDGPU=1"
         if [ "$AOMP_STANDALONE_BUILD" == 1 ]; then
           ASAN_CMAKE_OPTS="$ASAN_CMAKE_OPTS -DCMAKE_PREFIX_PATH=$AOMP_INSTALL_DIR/lib/llvm/lib/asan/cmake;$AOMP_INSTALL_DIR/lib/llvm/lib/cmake $AOMP_ASAN_ORIGIN_RPATH"
         else
           ASAN_CMAKE_OPTS="$ASAN_CMAKE_OPTS -DCMAKE_PREFIX_PATH=$ROCM_CMAKECONFIG_PATH;$INSTALL_PREFIX/lib/llvm/lib/asan $OPENMP_EXTRAS_ORIGIN_RPATH"
         fi
         echo " -----Running offload cmake for debug-asan ---- "
         mkdir -p $BUILD_DIR/build/offload_debug/asan
         cd $BUILD_DIR/build/offload_debug/asan
         echo ${AOMP_CMAKE} $ASAN_CMAKE_OPTS -DCMAKE_C_FLAGS="'$ASAN_FLAGS'" -DCMAKE_CXX_FLAGS="'$ASAN_FLAGS'" -DOFFLOAD_LIBDIR_SUFFIX="-debug/asan" $AOMP_REPOS/$AOMP_PROJECT_REPO_NAME/offload
         ${AOMP_CMAKE} $ASAN_CMAKE_OPTS -DCMAKE_C_FLAGS="'$ASAN_FLAGS'" -DCMAKE_CXX_FLAGS="'$ASAN_FLAGS'" -DOFFLOAD_LIBDIR_SUFFIX="-debug/asan" $AOMP_REPOS/$AOMP_PROJECT_REPO_NAME/offload
         if [ $? != 0 ] ; then
            echo "ERROR offload debug cmake failed. Cmake flags"
            echo "      $ASAN_CMAKE_OPTS"
            exit 1
         fi
      fi
   fi
fi
if [ "$1" != "install" ] ; then
if [ "$AOMP_LEGACY_OPENMP" == "1" ] && [ "$SANITIZER" != 1 ] ; then
  cd $BUILD_DIR/build/offload
  echo " -----Running $AOMP_NINJA_BIN for $BUILD_DIR/build/offload ---- "
  $AOMP_NINJA_BIN -j $AOMP_JOB_THREADS
  if [ $? != 0 ] ; then
        echo " "
        echo "ERROR: $AOMP_NINJA_BIN -j $AOMP_JOB_THREADS  FAILED"
        echo "To restart:"
        echo "  cd $BUILD_DIR/build/offload"
        echo "  $AOMP_NINJA_BIN"
        exit 1
  fi
fi

if [ "$AOMP_BUILD_SANITIZER" == 1 ] ; then
   cd $BUILD_DIR/build/offload/asan
   echo " -----Running $AOMP_NINJA_BIN for $BUILD_DIR/build/offload/asan ---- "
   $AOMP_NINJA_BIN -j $AOMP_JOB_THREADS
   if [ $? != 0 ] ; then
      echo " "
      echo "ERROR: $AOMP_NINJA_BIN -j $AOMP_JOB_THREADS  FAILED"
      echo "To restart:"
      echo "  cd $BUILD_DIR/build/offload/asan"
      echo "  $AOMP_NINJA_BIN"
      exit 1
   fi
fi

if [ "$AOMP_BUILD_PERF" == "1" ] ; then
   cd $BUILD_DIR/build/offload_perf
   echo
   echo
   echo " -----Running $AOMP_NINJA_BIN for $BUILD_DIR/build/offload_perf ---- "
   $AOMP_NINJA_BIN -j $AOMP_JOB_THREADS
   if [ $? != 0 ] ; then 
         echo "ERROR $AOMP_NINJA_BIN -j $AOMP_JOB_THREADS failed"
         exit 1
   fi
   if [ "$AOMP_BUILD_SANITIZER" == 1 ] ; then
      cd $BUILD_DIR/build/offload_perf/asan
      echo
      echo
      echo " ----- Running $AOMP_NINJA_BIN for $BUILD_DIR/build/offload_perf/asan ----- "
      $AOMP_NINJA_BIN -j $AOMP_JOB_THREADS
      if [ $? != 0 ] ; then
         echo "ERROR $AOMP_NINJA_BIN -j $AOMP_JOB_THREADS failed"
         exit 1
      fi
   fi
fi

if [ "$AOMP_BUILD_DEBUG" == "1" ] ; then
   if [ "$SANITIZER" != 1 ] ; then
      cd $BUILD_DIR/build/offload_debug
      echo
      echo
      echo " -----Running $AOMP_NINJA_BIN for $BUILD_DIR/build/offload_debug ---- "
      $AOMP_NINJA_BIN -j $AOMP_JOB_THREADS
      if [ $? != 0 ] ; then
         echo "ERROR $AOMP_NINJA_BIN -j $AOMP_JOB_THREADS failed"
         exit 1
      fi
   fi
   if [ "$AOMP_BUILD_SANITIZER" == 1 ] ; then
      cd $BUILD_DIR/build/offload_debug/asan
      echo
      echo
      echo " -----Running $AOMP_NINJA_BIN for $BUILD_DIR/build/offload_debug/asan ---- "
      $AOMP_NINJA_BIN -j $AOMP_JOB_THREADS
      if [ $? != 0 ] ; then
         echo "ERROR $AOMP_NINJA_BIN -j $AOMP_JOB_THREADS failed"
         exit 1
      fi
   fi
fi

   echo
   echo "Successful build of ./build_offload.sh .  Please run:"
   echo "  ./build_offload.sh install "
   echo
fi

#  ----------- Install only if asked  ----------------------------
if [ "$1" == "install" ] ; then
   clang_major=$("$AOMP_INSTALL_DIR"/bin/clang --version | grep -oP '(?<=clang version )[0-9]+')
   llvm_dylib=$(readlink "$AOMP_INSTALL_DIR"/lib/libLLVM.so)
   if [ "$AOMP_LEGACY_OPENMP" == "1" ] && [ "$SANITIZER" != 1 ] ; then
      cd $BUILD_DIR/build/offload
      echo
      echo " -----Installing to $INSTALL_OFFLOAD/lib ----- "
      $SUDO $AOMP_NINJA_BIN -j $AOMP_JOB_THREADS install
      if [ $? != 0 ] ; then
         echo "ERROR $AOMP_NINJA_BIN install failed "
         exit 1
      fi
   fi

   if [ "$AOMP_BUILD_SANITIZER" == 1 ] ; then
      cd $BUILD_DIR/build/offload/asan
      echo
      echo " -----Installing to $INSTALL_OFFLOAD/lib/asan ----- "
      $SUDO $AOMP_NINJA_BIN -j $AOMP_JOB_THREADS install
      if [ $? != 0 ] ; then
         echo "ERROR $AOMP_NINJA_BIN install failed "
         exit 1
      fi
   fi

   if [ "$AOMP_BUILD_PERF" == "1" ]; then
     cd $BUILD_DIR/build/offload_perf
     echo
     echo " -----Installing to $INSTALL_OFFLOAD/lib-perf ----- "
     $SUDO $AOMP_NINJA_BIN -j $AOMP_JOB_THREADS install
     if [ $? != 0 ] ; then
        echo "ERROR $AOMP_NINJA_BIN install failed "
        exit 1
     fi
     if [ ! -h $AOMP_INSTALL_DIR/lib-perf/$llvm_dylib ] && [ "$llvm_dylib" != "" ]; then
       cd $AOMP_INSTALL_DIR/lib-perf
       ln -s ../lib/$llvm_dylib $llvm_dylib
     fi

     if [ "$AOMP_BUILD_SANITIZER" == 1 ] ; then
        cd $BUILD_DIR/build/offload_perf/asan
        echo
        echo " ----- Installing to $INSTALL_OFFLOAD/lib-perf/asan ----- "
        $SUDO $AOMP_NINJA_BIN -j $AOMP_JOB_THREADS install
        if [ $? != 0 ] ; then
           echo "ERROR $AOMP_NINJA_BIN install failed "
           exit 1
        fi
     fi
   fi

   if [ "$AOMP_BUILD_DEBUG" == "1" ] ; then
      _ompd_dir="$AOMP_INSTALL_DIR/lib-debug/ompd"
      #  This is the new locationof the ompd directory
      [[ ! -d $_ompd_dir ]] && _ompd_dir="$AOMP_INSTALL_DIR/share/gdb/python/ompd"
      if [ "$SANITIZER" != 1 ] ; then
         cd $BUILD_DIR/build/offload_debug
         echo
         echo " -----Installing to $INSTALL_OFFLOAD/lib-debug ---- "
         $SUDO $AOMP_NINJA_BIN -j $AOMP_JOB_THREADS install
         if [ $? != 0 ] ; then
            echo "ERROR $AOMP_NINJA_BIN install failed "
            exit 1
         fi
         if [ ! -h $AOMP_INSTALL_DIR/lib-debug/$llvm_dylib ] && [ "$llvm_dylib" != "" ]; then
            cd $AOMP_INSTALL_DIR/lib-debug
            ln -s ../lib/$llvm_dylib $llvm_dylib
         fi
      fi
      if [ "$AOMP_BUILD_SANITIZER" == 1 ] ; then
         cd $BUILD_DIR/build/offload_debug/asan
         echo " -----Installing to $INSTALL_OFFLOAD/lib-debug/asan ---- "
         $SUDO $AOMP_NINJA_BIN -j $AOMP_JOB_THREADS install
         if [ $? != 0 ] ; then
            echo "ERROR $AOMP_NINJA_BIN install failed "
            exit 1
         fi
      fi
   fi
   # we do not yet have OMPD in llvm 12, disable this for now.
   # Copy selected debugable runtime sources into the installation $ompd_dir/src directory
   # to satisfy the above -fdebug-prefix-map.
   if [ "$AOMP_STANDALONE_BUILD" == 1 ]; then
      $SUDO mkdir -p $_ompd_dir/src/offload/src
      echo cp -rp $AOMP_REPOS/$AOMP_PROJECT_REPO_NAME/offload/src $_ompd_dir/src/offload
      $SUDO cp -rp $AOMP_REPOS/$AOMP_PROJECT_REPO_NAME/offload/src $_ompd_dir/src/offload
   else
     $SUDO mkdir -p $AOMP_INSTALL_DIR/lib-debug/src/offload/src
     echo cp -rp $LLVM_PROJECT_ROOT/offload/src $AOMP_INSTALL_DIR/lib-debug/src/offload
     $SUDO cp -rp $LLVM_PROJECT_ROOT/offload/src $AOMP_INSTALL_DIR/lib-debug/src/offload
   fi
fi

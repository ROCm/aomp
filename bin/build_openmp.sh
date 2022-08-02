#!/bin/bash
#
#  build_openmp.sh:  Script to build the AOMP runtime libraries and debug libraries.  
#                This script will install in location defined by AOMP env variable
#

# --- Start standard header to set AOMP environment variables ----
realpath=`realpath $0`
thisdir=`dirname $realpath`
. $thisdir/aomp_common_vars
# --- end standard header ----

INSTALL_OPENMP=${INSTALL_OPENMP:-$AOMP_INSTALL_DIR}

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
      echo "        Please install CUDA to build openmp"
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

export LLVM_DIR=$AOMP_INSTALL_DIR
GFXSEMICOLONS=`echo $GFXLIST | tr ' ' ';' `
ALTAOMP=${ALTAOMP:-$AOMP}
COMMON_CMAKE_OPTS="-DOPENMP_ENABLE_LIBOMPTARGET=1
-DOPENMP_ENABLE_LIBOMPTARGET_HSA=1
-DCMAKE_INSTALL_PREFIX=$INSTALL_OPENMP
-DOPENMP_TEST_C_COMPILER=$AOMP/bin/clang
-DOPENMP_TEST_CXX_COMPILER=$AOMP/bin/clang++
-DCMAKE_C_COMPILER=$ALTAOMP/bin/clang
-DCMAKE_CXX_COMPILER=$ALTAOMP/bin/clang++
-DLIBOMPTARGET_AMDGCN_GFXLIST=$GFXSEMICOLONS
-DDEVICELIBS_ROOT=$DEVICELIBS_ROOT
-DLIBOMP_COPY_EXPORTS=OFF
-DLIBOMPTARGET_ENABLE_DEBUG=ON
-DAOMP_STANDALONE_BUILD=$AOMP_STANDALONE_BUILD
-DLLVM_DIR=$LLVM_DIR"


if [ "$AOMP_STANDALONE_BUILD" == 0 ]; then
  COMMON_CMAKE_OPTS="$COMMON_CMAKE_OPTS
  -DLLVM_MAIN_INCLUDE_DIR=$LLVM_PROJECT_ROOT/llvm/include
  -DLIBOMPTARGET_LLVM_INCLUDE_DIRS=$LLVM_PROJECT_ROOT/llvm/include
  -DENABLE_DEVEL_PACKAGE=ON -DENABLE_RUN_PACKAGE=ON"
else
  COMMON_CMAKE_OPTS="$COMMON_CMAKE_OPTS \
-DLLVM_MAIN_INCLUDE_DIR=$AOMP_REPOS/$AOMP_PROJECT_REPO_NAME/llvm/include \
-DLIBOMPTARGET_LLVM_INCLUDE_DIRS=$AOMP_REPOS/$AOMP_PROJECT_REPO_NAME/llvm/include \
-DLIBOMP_USE_HWLOC=ON -DLIBOMP_HWLOC_INSTALL_DIR=$AOMP_SUPP/hwloc"
fi

if [ "$AOMP_BUILD_CUDA" == 1 ] ; then
   COMMON_CMAKE_OPTS="$COMMON_CMAKE_OPTS
-DLIBOMPTARGET_NVPTX_ENABLE_BCLIB=ON
-DLIBOMPTARGET_NVPTX_CUDA_COMPILER=$AOMP/bin/clang++
-DLIBOMPTARGET_NVPTX_BC_LINKER=$AOMP/bin/llvm-link
-DLIBOMPTARGET_NVPTX_COMPUTE_CAPABILITIES=$NVPTXGPUS"
else
#  Need to force CUDA off this way in case cuda is installed in this system
   COMMON_CMAKE_OPTS="$COMMON_CMAKE_OPTS
-DCUDA_TOOLKIT_ROOT_DIR=OFF"
fi

# Enable OpenMP Sanitizer Build
if [ "$AOMP_BUILD_CUDA" != 1 ] && [ "$AOMP_BUILD_SANITIZER" == "ON" ]; then
	COMMON_CMAKE_OPTS+=" -DSANITIZER_AMDGPU=1"
fi

# This is how we tell the hsa plugin where to find hsa
export HSA_RUNTIME_PATH=$ROCM_DIR

#breaks build as it cant find rocm-path
#export HIP_DEVICE_LIB_PATH=$ROCM_DIR/lib

if [ "$1" != "nocmake" ] && [ "$1" != "install" ] ; then 

   echo " " 
   echo "This is a FRESH START. ERASING any previous builds in $BUILD_DIR/openmp "
   echo "Use ""$0 nocmake"" or ""$0 install"" to avoid FRESH START."
      echo rm -rf $BUILD_DIR/build/openmp
      rm -rf $BUILD_DIR/build/openmp
      MYCMAKEOPTS="$COMMON_CMAKE_OPTS -DCMAKE_BUILD_TYPE=Release $AOMP_ORIGIN_RPATH -DROCM_DIR=$ROCM_DIR"
      mkdir -p $BUILD_DIR/build/openmp
      cd $BUILD_DIR/build/openmp
      echo " -----Running openmp cmake ---- " 
      echo ${AOMP_CMAKE} $MYCMAKEOPTS  $AOMP_REPOS/$AOMP_PROJECT_REPO_NAME/openmp
      ${AOMP_CMAKE} $MYCMAKEOPTS  $AOMP_REPOS/$AOMP_PROJECT_REPO_NAME/openmp
      if [ $? != 0 ] ; then 
         echo "ERROR openmp cmake failed. Cmake flags"
         echo "      $MYCMAKEOPTS"
         exit 1
      fi

   if [ "$AOMP_BUILD_DEBUG" == "1" ] ; then
      echo rm -rf $BUILD_DIR/build/openmp_debug
      rm -rf $BUILD_DIR/build/openmp_debug
      
      MYCMAKEOPTS="$COMMON_CMAKE_OPTS \
-DLIBOMPTARGET_NVPTX_DEBUG=ON \
-DCMAKE_BUILD_TYPE=Debug \
$AOMP_ORIGIN_RPATH \
-DROCM_DIR=$ROCM_DIR \
-DLIBOMP_ARCH=x86_64 \
-DLIBOMP_OMPT_SUPPORT=ON \
-DLIBOMP_USE_DEBUGGER=ON \
-DLIBOMP_CPPFLAGS='-O0' \
-DLIBOMP_OMPD_SUPPORT=ON \
-DLIBOMP_OMPT_DEBUG=ON \
-DCMAKE_CXX_FLAGS=-g -DCMAKE_C_FLAGS=-g \
-DOPENMP_SOURCE_DEBUG_MAP="\""-fdebug-prefix-map=$AOMP_REPOS/$AOMP_PROJECT_REPO_NAME/openmp=$AOMP_INSTALL_DIR/lib-debug/src/openmp"\"" "

      # The 'pip install --system' command is not supported on non-debian systems. This will disable
      # the system option if the debian_version file is not present.
      if [ ! -f /etc/debian_version ]; then
         echo "==> Non-Debian OS, disabling use of pip install --system"
         MYCMAKEOPTS="$MYCMAKEOPTS -DDISABLE_SYSTEM_NON_DEBIAN=1"
      fi

      # Redhat 7.6 does not have python36-devel package, which is needed for ompd compilation.
      # This is acquired through RH Software Collections.
      if [ -f /opt/rh/rh-python36/enable ]; then
         echo "==> Using python3.6 out of rh tools."
         MYCMAKEOPTS="$MYCMAKEOPTS -DPython3_ROOT_DIR=/opt/rh/rh-python36/root/bin -DPYTHON_HEADERS=/opt/rh/rh-python36/root/usr/include/python3.6m"
      fi

      mkdir -p $BUILD_DIR/build/openmp_debug
      cd $BUILD_DIR/build/openmp_debug
      echo
      echo " -----Running openmp cmake for debug ---- " 
      echo ${AOMP_CMAKE} $MYCMAKEOPTS  $AOMP_REPOS/$AOMP_PROJECT_REPO_NAME/openmp
      ${AOMP_CMAKE} $MYCMAKEOPTS  $AOMP_REPOS/$AOMP_PROJECT_REPO_NAME/openmp
      if [ $? != 0 ] ; then 
         echo "ERROR openmp debug cmake failed. Cmake flags"
         echo "      $MYCMAKEOPTS"
         exit 1
      fi
   fi
fi

cd $BUILD_DIR/build/openmp
echo " -----Running make for $BUILD_DIR/build/openmp ---- "
make -j $AOMP_JOB_THREADS
if [ $? != 0 ] ; then 
      echo " "
      echo "ERROR: make -j $AOMP_JOB_THREADS  FAILED"
      echo "To restart:" 
      echo "  cd $BUILD_DIR/build/openmp"
      echo "  make"
      exit 1
fi

if [ "$AOMP_BUILD_DEBUG" == "1" ] ; then
   cd $BUILD_DIR/build/openmp_debug
   echo
   echo
   echo " -----Running make for $BUILD_DIR/build/openmp_debug ---- "
   make -j $AOMP_JOB_THREADS
   if [ $? != 0 ] ; then 
         echo "ERROR make -j $AOMP_JOB_THREADS failed"
         exit 1
   fi
fi

if [ "$1" != "install" ] ; then
   echo
   echo "Successful build of ./build_openmp.sh .  Please run:"
   echo "  ./build_openmp.sh install "
   echo
fi

#  ----------- Install only if asked  ----------------------------
if [ "$1" == "install" ] ; then 
   cd $BUILD_DIR/build/openmp
   echo
   echo " -----Installing to $INSTALL_OPENMP/lib ----- "
   $SUDO make install
   if [ $? != 0 ] ; then
      echo "ERROR make install failed "
      exit 1
   fi

   if [ "$AOMP_BUILD_DEBUG" == "1" ] ; then
      cd $BUILD_DIR/build/openmp_debug
      echo
      echo " -----Installing to $INSTALL_OPENMP/lib-debug ---- " 
      $SUDO make install 
      if [ $? != 0 ] ; then 
         echo "ERROR make install failed "
         exit 1
      fi

      # Rename ompd module library. This was introduced with the Python3 build. The include
      # of ompdModule would fail otherwise.
      if [ -f "$AOMP_INSTALL_DIR/lib-debug/ompd/ompdModule.cpython-36m-x86_64-linux-gnu.so" ]; then
        if [ -f "$AOMP_INSTALL_DIR/lib-debug/ompd/ompdModule.so" ]; then
          echo "==> Removing old ompdModule.so"
          rm "$AOMP_INSTALL_DIR/lib-debug/ompd/ompdModule.so"
        fi
        echo "==> Renaming ompdModule.cpython-36m-x86_64-linux-gnu.so to ompdModule.so"
        mv $AOMP_INSTALL_DIR/lib-debug/ompd/ompdModule.cpython-36m-x86_64-linux-gnu.so $AOMP_INSTALL_DIR/lib-debug/ompd/ompdModule.so
      fi
      if [[ "$DEVEL_PACKAGE" =~ "devel" ]]; then
        AOMP_INSTALL_DIR="$AOMP_INSTALL_DIR/""$DEVEL_PACKAGE"
        echo "Request for devel package found."
      fi

      # we do not yet have OMPD in llvm 12, disable this for now.
      # Copy selected debugable runtime sources into the installation lib-debug/src directory
      # to satisfy the above -fdebug-prefix-map.
      $SUDO mkdir -p $AOMP_INSTALL_DIR/lib-debug/src/openmp/runtime/src
      echo cp -rp $AOMP_REPOS/$AOMP_PROJECT_REPO_NAME/openmp/runtime/src $AOMP_INSTALL_DIR/lib-debug/src/openmp/runtime
      $SUDO cp -rp $AOMP_REPOS/$AOMP_PROJECT_REPO_NAME/openmp/runtime/src $AOMP_INSTALL_DIR/lib-debug/src/openmp/runtime
      $SUDO mkdir -p $AOMP_INSTALL_DIR/lib-debug/src/openmp/libomptarget/src
      echo cp -rp $AOMP_REPOS/$AOMP_PROJECT_REPO_NAME/openmp/libomptarget/src $AOMP_INSTALL_DIR/lib-debug/src/openmp/libomptarget
      $SUDO cp -rp $AOMP_REPOS/$AOMP_PROJECT_REPO_NAME/openmp/libomptarget/src $AOMP_INSTALL_DIR/lib-debug/src/openmp/libomptarget
      echo cp -rp $AOMP_REPOS/$AOMP_PROJECT_REPO_NAME/openmp/libomptarget/plugins $AOMP_INSTALL_DIR/lib-debug/src/openmp/libomptarget
      $SUDO cp -rp $AOMP_REPOS/$AOMP_PROJECT_REPO_NAME/openmp/libomptarget/plugins $AOMP_INSTALL_DIR/lib-debug/src/openmp/libomptarget
      $SUDO mkdir -p $AOMP_INSTALL_DIR/lib-debug/src/openmp/libompd/src
      $SUDO cp -rp $AOMP_REPOS/$AOMP_PROJECT_REPO_NAME/openmp/libompd/src $AOMP_INSTALL_DIR/lib-debug/src/openmp/libompd
   fi
fi

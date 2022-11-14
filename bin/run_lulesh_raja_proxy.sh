#!/bin/bash

function usage(){
  echo ""
  echo "------------ Usage ------------"
  echo "./run_rajaperf.sh [backend] [option]"
  echo "Backends: hip, openmp"
  echo "Options:  build, perf"
  echo "------------------------------"
  echo ""
}

# --- Start standard header to set AOMP environment variables ----
realpath=`realpath $0`
thisdir=`dirname $realpath`
. $thisdir/aomp_common_vars
# --- end standard header ----

# Setup AOMP variables
AOMP=${AOMP:-/usr/lib/aomp}
ROCM_DIR=${ROCM_DIR:-}

# Use function to set and test AOMP_GPU
setaompgpu


if [ "$1" == "hip" ]; then
  BUILD_SUFFIX=hip
else
  BUILD_SUFFIX=omptarget
fi

# Begin configuration
pushd $AOMP_REPOS_TEST/RAJAProxies
cd $AOMP_REPOS_TEST/RAJAProxies

export LD_LIBRARY_PATH=opt/rocm//lib:opt/rocm//hsa/lib:opt/rocm/hip/lib:$LD_LIBRARY_PATH
echo $LD_LIBRARY_PATH

if [ "$2" == "build" ]; then

rm -rf build_${BUILD_SUFFIX}
mkdir build_${BUILD_SUFFIX}
pushd build_${BUILD_SUFFIX}

  if [ "$1" == "hip" ]; then
   $AOMP_CMAKE \
      -DCMAKE_BUILD_TYPE=Release \
      -DCMAKE_CXX_COMPILER=${AOMP}/bin/clang++ \
      -DCMAKE_C_COMPILER=${AOMP}/bin/clang \
      -DENABLE_CUDA=Off \
      -DENABLE_HIP=On \
      -DENABLE_ALL_WARNINGS=Off \
      -DCMAKE_INSTALL_PREFIX=../install_${BUILD_SUFFIX} \
      -DENABLE_TESTS=Off \
      "$@" \
      ..

  elif [ "$1" == "openmp" ]; then

  $AOMP_CMAKE \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_CXX_COMPILER=${AOMP}/bin/clang++ \
    -DCMAKE_C_COMPILER=${AOMP}/bin/clang \
    -DENABLE_CUDA=Off \
    -DENABLE_HIP=Off \
    -DENABLE_OPENMP=On \
    -DENABLE_TARGET_OPENMP=On \
    -DRAJA_ENABLE_TARGET_OPENMP=On \
    -DCMAKE_CXX_FLAGS="-DENABLE_TARGET_OPENMP" \
    -DOpenMP_CXX_FLAGS="-fopenmp=libomp;-fopenmp-targets=amdgcn-amd-amdhsa;-Xopenmp-target=amdgcn-amd-amdhsa;-march=$AOMP_GPU" \
    -DENABLE_ALL_WARNINGS=Off \
    -DCMAKE_INSTALL_PREFIX=../install_${BUILD_SUFFIX} \
    -DENABLE_TESTS=Off \
    "$@" \
    ..

  $AOMP_CMAKE \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_CXX_COMPILER=${AOMP}/bin/clang++ \
    -DCMAKE_C_COMPILER=${AOMP}/bin/clang \
    -DENABLE_CUDA=Off \
    -DENABLE_HIP=Off \
    -DENABLE_OPENMP=On \
    -DENABLE_TARGET_OPENMP=On \
    -DRAJA_ENABLE_TARGET_OPENMP=On \
    -DCMAKE_CXX_FLAGS="-DENABLE_TARGET_OPENMP" \
    -DOpenMP_CXX_FLAGS="-fopenmp=libomp;-fopenmp-targets=amdgcn-amd-amdhsa;-Xopenmp-target=amdgcn-amd-amdhsa;-march=$AOMP_GPU" \
    -DENABLE_ALL_WARNINGS=Off \
    -DCMAKE_INSTALL_PREFIX=../install_${BUILD_SUFFIX} \
    -DENABLE_TESTS=Off \
    "$@" \
    ..

  else
    echo "Option $2 not supported. Please choose from $build_targets"
    usage
  fi
  pwd
  echo "calling make ...."
  make -j $AOMP_JOB_THREADS
  make install
  popd
  exit 0
fi
  pwd
  if [ -d build_${BUILD_SUFFIX} ] && [ "$2" != "build" ]; then
    if [ "$2" == "perf" ]; then
      if [ "$1" == "hip" ]; then
        build_${BUILD_SUFFIX}/bin/raja-perf.exe --show-progress --refvar Base_HIP
      elif [ "$1" == "openmp" ] ; then
        if [ ! -f build_${BUILD_SUFFIX}/bin/lulesh-v1.0-RAJA-omp.exe ] ; then 
          echo "ERROR file build_${BUILD_SUFFIX}/bin/lulesh-v1.0-RAJA-omp.exe not found"
          echo "      please build raja first"
	        exit 1
        fi
        build_${BUILD_SUFFIX}/bin/lulesh-v1.0-RAJA-seq.exe 
        build_${BUILD_SUFFIX}/bin/lulesh-v1.0-RAJA-omp.exe

        build_${BUILD_SUFFIX}/bin/lulesh-v2.0-RAJA-seq.exe
        build_${BUILD_SUFFIX}/bin/lulesh-v2.0-RAJA-omp.exe
      fi
    else
      echo Error: $2 not a recognized option.
      usage
      exit 1
    fi
  else
    echo Error: No build directory found!
    usage
    exit 1
fi

#!/bin/bash

# --- Start standard header to set AOMP environment variables ----
realpath=`realpath $0`
thisdir=`dirname $realpath`
export AOMP_USE_CCACHE=0

. $thisdir/aomp_common_vars
# --- end standard header ----

function usage(){
  echo ""
  echo "------------ Usage ------------"
  echo "./run_rajaperf.sh [backend] [option]"
  echo "Backends: hip, openmp"
  echo "Options:  build, unit, perf"
  echo "------------------------------"
  echo ""
}

# Setup AOMP variables
AOMP=${AOMP:-/usr/lib/aomp}

# Use function to set and test AOMP_GPU
setaompgpu

# Check cmake version
cmake_regex="(([0-9])+\.([0-9]+)\.[0-9]+)"
cmake_ver_str=$($AOMP_CMAKE  --version)
if [[ "$cmake_ver_str" =~ $cmake_regex ]]; then
  cmake_ver=${BASH_REMATCH[1]}
  cmake_major_ver=${BASH_REMATCH[2]}
  cmake_minor_ver=${BASH_REMATCH[3]}
  echo "Cmake found: version $cmake_ver"
else
  echo "ERROR: No cmake found, exiting..."
  return 1
fi


if [ "$1" == "hip" ]; then
  BUILD_SUFFIX=hip
else
  BUILD_SUFFIX=omptarget
fi

build_targets="hip openmp"
if [ "$2" == "build" ]; then
  # Begin configuration
  pushd $AOMP_REPOS_TEST/RAJAPerf
  git reset --hard 73f73cfa1979bc9944d9f34ee0059873fb38d79d
  git submodule update --recursive
  
  rm -rf build_${BUILD_SUFFIX}
  mkdir build_${BUILD_SUFFIX}
  pushd build_${BUILD_SUFFIX}

  if [ "$1" == "hip" ]; then
    $AOMP/bin/clang --version | grep AOMP
    if [ $? -eq 0 ]; then
      export HIP_PATH="$AOMP"
      export HIP_CLANG_PATH=$AOMP/bin
    else
      export HIP_PATH="$AOMP/../"
      export HIP_CLANG_PATH=$AOMP/bin
    fi
    export ROCM_PATH=$HIP_PATH
    $AOMP_CMAKE \
      -DCMAKE_BUILD_TYPE=Release \
      -DCMAKE_CXX_COMPILER=${AOMP}/bin/clang++ \
      -DENABLE_CUDA=Off \
      -DENABLE_HIP=On \
      -DCMAKE_PREFIX_PATH="$AOMP;$AOMP/..;/opt/rocm" \
      -DENABLE_ALL_WARNINGS=Off \
      -DCMAKE_INSTALL_PREFIX=../install_${BUILD_SUFFIX} \
      -DCMAKE_HIP_ARCHITECTURES=$AOMP_GPU \
      -DENABLE_TESTS=On \
      "$@" \
      ..
  elif [ "$1" == "openmp" ]; then
    $AOMP_CMAKE \
      -DCMAKE_FIND_DEBUG_MODE=OFF \
      -DCMAKE_BUILD_TYPE=Release \
      -DCMAKE_CXX_COMPILER=${AOMP}/bin/clang++ \
      -DENABLE_CUDA=OFF \
      -DRAJA_ENABLE_CUDA=OFF \
      -DENABLE_HIP=OFF \
      -DRAJA_ENABLE_HIP=OFF \
      -DENABLE_OPENMP=ON \
      -DRAJA_ENABLE_OPENMP=ON \
      -DENABLE_TARGET_OPENMP=ON \
      -DRAJA_ENABLE_TARGET_OPENMP=ON \
      -DCMAKE_CXX_FLAGS="-O3 -DENABLE_TARGET_OPENMP -fopenmp=libomp -fopenmp-targets=amdgcn-amd-amdhsa -Xopenmp-target=amdgcn-amd-amdhsa -march=$AOMP_GPU" \
      -DROCM_ARCH=gfx90a \
      -DENABLE_ALL_WARNINGS=OFF \
      -DCMAKE_INSTALL_PREFIX=../install_${BUILD_SUFFIX} \
      -DENABLE_TESTS=ON \
      "$@" \
      ..
  else
    echo "Option $2 not supported. Please choose from $build_targets"
    usage
  fi

  LESS_THREADS=$(( AOMP_JOB_THREADS/2 ))
  LESS_THREADS=$(( $LESS_THREADS > 32 ? 32 : $LESS_THREADS ))
  MAKE_THREADS=${MAKE_THREADS:-$LESS_THREADS}
  echo "Using $MAKE_THREADS threads for make."
  make -j $MAKE_THREADS
  # Do not continue if build fails
  if [ $? != 0 ]; then
    echo "ERROR: Make returned non-zero, exiting..."
    exit 1
  fi
  
  popd
  exit 0
fi

# Run performance tests or unit tests
pushd $AOMP_REPOS_TEST/RAJAPerf
if [ -d build_${BUILD_SUFFIX} ] && [ "$2" != "build" ]; then
  if [ "$2" == "perf" ]; then
    if [ "$1" == "hip" ]; then
      build_${BUILD_SUFFIX}/bin/raja-perf.exe --show-progress --refvar Base_HIP
    elif [ "$1" == "openmp" ] ; then
      if [ ! -f build_${BUILD_SUFFIX}/bin/raja-perf-omptarget.exe ] ; then
        echo "ERROR file build_${BUILD_SUFFIX}/bin/raja-perf-omptarget.exe not found"
        echo "      please build raja first"
	exit 1
      fi
      build_${BUILD_SUFFIX}/bin/raja-perf-omptarget.exe --show-progress --refvar Base_OMPTarget
    fi
  elif [ "$2" == "unit" ]; then
      cd build_${BUILD_SUFFIX}
      make test
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
popd

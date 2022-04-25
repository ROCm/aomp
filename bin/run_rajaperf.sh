#!/bin/bash
# --- Start standard header to set build environment variables ----
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
[ ! -L "$0" ] || thisdir=$(getdname `readlink -f "$0"`)
. $thisdir/aomp_common_vars

function usage(){
  echo ""
  echo "------------ Usage ------------"
  echo "./run_rajaperf.sh [backend] [option]"
  echo "Backends: hip, openmp"
  echo "Options:  build, unit, perf"
  echo "------------------------------"
  echo ""
}

function cmake_warning(){
  echo "-----------------------------------------------------------------------------------------------------"
  echo "Warning!! It is recommended to build Raja with a cmake version between 3.9 and and 3.16.8 (inclusive)."
  echo "Raja Performance Suite may fail to build otherwise."
  echo "HIT ENTER TO CONTINUE or CTRL-C TO CANCEL"
  echo "-----------------------------------------------------------------------------------------------------"
  read
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
  if [ "$cmake_major_ver" != "3" ]; then
    cmake_warning
  elif (( $(echo "$cmake_minor_ver > 16"| bc -l) ||  $(echo "$cmake_minor_ver < 9" | bc -l) )); then
    cmake_warning
  fi
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
  git reset --hard 43b8ad43
  git submodule update --recursive
  # Apply patches
  patchrepo $AOMP_REPOS_TEST/RAJAPerf/tpl/RAJA
  cd $AOMP_REPOS_TEST/RAJAPerf
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
    $AOMP_CMAKE \
      -DCMAKE_BUILD_TYPE=Release \
      -DCMAKE_CXX_COMPILER=${AOMP}/bin/clang++ \
      -DENABLE_CUDA=Off \
      -DENABLE_HIP=On \
      -DCMAKE_PREFIX_PATH="$AOMP;$AOMP/..;/opt/rocm" \
      -DENABLE_ALL_WARNINGS=Off \
      -DCMAKE_INSTALL_PREFIX=../install_${BUILD_SUFFIX} \
      -DENABLE_TESTS=On \
      "$@" \
      ..
  elif [ "$1" == "openmp" ]; then
    $AOMP_CMAKE \
      -DCMAKE_BUILD_TYPE=Release \
      -DCMAKE_CXX_COMPILER=${AOMP}/bin/clang++ \
      -DENABLE_CUDA=Off \
      -DENABLE_OPENMP=On \
      -DENABLE_TARGET_OPENMP=On \
      -DCMAKE_CXX_FLAGS="-DENABLE_TARGET_OPENMP" \
      -DROCM_ARCH=gfx90a \
      -DOpenMP_CXX_FLAGS="-fopenmp;-fopenmp-targets=amdgcn-amd-amdhsa;-Xopenmp-target=amdgcn-amd-amdhsa;-march=$AOMP_GPU" \
      -DENABLE_ALL_WARNINGS=Off \
      -DCMAKE_INSTALL_PREFIX=../install_${BUILD_SUFFIX} \
      -DENABLE_TESTS=On \
      "$@" \
      ..
  else
    echo "Option $2 not supported. Please choose from $build_targets"
    usage
  fi
  make -j $AOMP_JOB_THREADS
  # Do not continue if build fails
  if [ $? != 0 ]; then
    echo "ERROR: Make returned non-zero, exiting..."
    exit 1
  fi
  # Cleanup patches
  removepatch $AOMP_REPOS_TEST/RAJAPerf/tpl/RAJA
  popd
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

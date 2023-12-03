#!/bin/bash

# run_rushlarsen.sh - runs rush_larsen microbenchmarks in the $AOMP_REPOS_TEST dir.
# User can set RUN_OPTIONS to control what variants(openmp, hip) are selected.

# --- Start standard header to set AOMP environment variables ----
realpath=`realpath $0`
thisdir=`dirname $realpath`
export AOMP_USE_CCACHE=0

. $thisdir/aomp_common_vars
# --- end standard header ----

# Setup AOMP variables
AOMP=${AOMP:-/usr/lib/aomp}
AOMPHIP=${AOMPHIP:-$AOMP}
FLANG=${FLANG:-flang}

# Use function to set and test AOMP_GPU
setaompgpu

RUN_OPTIONS=${RUN_OPTIONS:-"openmp hip fomp"}

# OMP options
omp_cxx="$AOMP/bin/clang++"
omp_flags="-O3 -fopenmp -fopenmp-targets=amdgcn-amd-amdhsa -Xopenmp-target=amdgcn-amd-amdhsa -march=$AOMP_GPU -DOMP -DOMP_TARGET_GPU -fopenmp-target-fast"
omp_dir="tests/rush_larsen/rush_larsen_gpu_omp"
omp_exec="rush_larsen_gpu_omp"

fomp_f90="$AOMP/bin/$FLANG"
fomp_flags="-O3 -fopenmp -fopenmp-targets=amdgcn-amd-amdhsa -Xopenmp-target=amdgcn-amd-amdhsa -march=$AOMP_GPU -DOMP -DOMP_TARGET_GPU -g"
fomp_dir="tests/rush_larsen/rush_larsen_gpu_omp_fort"
fomp_exec="rush_larsen_gpu_omp_fort"

# HIP Options
hip_cxx="$AOMPHIP/bin/hipcc"
hip_flags="-D__HIP_PLATFORM_AMD__ -I${AOMPHIP}/include -O3 --cuda-gpu-arch=${AOMP_GPU} --rocm-path=${AOMPHIP} -x hip"
hip_dir="tests/rush_larsen/rush_larsen_gpu_hip"
hip_exec="rush_larsen_gpu_hip"

if [ -d $AOMP_REPOS_TEST/goulash ]; then
  cd $AOMP_REPOS_TEST/goulash
  goulash_home=$AOMP_REPOS_TEST/goulash
  rm -f results.txt
  rm -f Lresults.txt
else
  echo "ERROR: goulash not found in $AOMP_REPOS_TEST."
  exit 1
fi

echo RUN_OPTIONS: $RUN_OPTIONS
for option in $RUN_OPTIONS; do
  if [ "$option" == "openmp" ]; then
    cd $goulash_home/$omp_dir
    make clean
    make CXX="$omp_cxx" CXXFLAGS="$omp_flags" COMPILERID="-DCOMPILERID=amd"
    if [ $? -ne 1 ]; then
      ./$omp_exec 100 10 2>&1 | tee -a $goulash_home/results.txt
      ./$omp_exec 100000 .00000001 2>&1 | tee -a $goulash_home/Lresults.txt
    fi
  elif [ "$option" == "hip" ]; then
    cd $goulash_home/$hip_dir
    make clean
    make CXX="$hip_cxx" CXXFLAGS="$hip_flags" COMPILERID="-DCOMPILERID=amd"
    if [ $? -ne 1 ]; then
      ./$hip_exec 100 10 2>&1 | tee -a $goulash_home/results.txt
      ./$hip_exec 100000 .00000001 2>&1 | tee -a $goulash_home/Lresults.txt
    fi
  elif [ "$option" == "fomp" ]; then
    cd $goulash_home/$fomp_dir
    make clean
    make FC="$fomp_f90" FCFLAGS="$fomp_flags" COMPILERID='-DCOMPILERID=\"amd\"'
    if [ $? -ne 1 ]; then
      ./$fomp_exec 100 10 2>&1 | tee -a $goulash_home/results.txt
      ./$fomp_exec 100000 .00000001 2>&1 | tee -a $goulash_home/Lresults.txt
    fi
  else
    echo "ERROR: Option not recognized: $option."
    exit 1
  fi
  echo >>  $goulash_home/results.txt
  echo >>  $goulash_home/Lresults.txt
done

#!/bin/bash

# run_XSBench.sh - runs XSBench in the $AOMP_REPOS_TEST dir.
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

# Use function to set and test AOMP_GPU
setaompgpu

RUN_OPTIONS=${RUN_OPTIONS:-"openmp hip"}
#omp_flags="-O3 -fopenmp -fopenmp-targets=amdgcn-amd-amdhsa -Xopenmp-target=amdgcn-amd-amdhsa -march=$AOMP_GPU -DOMP -DOMP_TARGET_GPU"
#hip_flags="-O3 --offload-arch=$AOMP_GPU -DHIP -x hip"
#omp_src="main.cpp OMPStream.cpp"
#hip_src="main.cpp HIPStream.cpp"
#std="-std=c++11"

if [ -d $AOMP_REPOS_TEST/XSBench ]; then
  cd $AOMP_REPOS_TEST/XSBench
  rm -f results.txt
else
  echo "ERROR: XSBench not found in $AOMP_REPOS_TEST."
  exit 1
fi

echo RUN_OPTIONS: $RUN_OPTIONS
for option in $RUN_OPTIONS; do
  if [ "$option" == "openmp" ]; then
    cd openmp-offload
    make clean
    export PATH=$AOMP/bin:$PATH
    make COMPILER=amd
    if [ $? -ne 1 ]; then
      ./XSBench -m event 2>&1 | tee -a results.txt
    fi
    cd ..
  elif [ "$option" == "hip" ]; then
    cd hip
    make clean
    export PATH=$AOMP/bin:$PATH
    make
    if [ $? -ne 1 ]; then
      ./XSBench -m event 2>&1 | tee -a results.txt
    fi
    cd ..
  else
    echo "ERROR: Option not recognized: $option."
    exit 1
  fi
  echo >> results.txt
done

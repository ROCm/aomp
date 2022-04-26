#!/bin/bash

# --- Start standard header to set AOMP environment variables ----
realpath=`realpath $0`
thisdir=`dirname $realpath`
. $thisdir/aomp_common_vars
# --- end standard header ----

# Setup AOMP variables
AOMP=${AOMP:-/usr/lib/aomp}

# Use function to set and test AOMP_GPU
setaompgpu

AOMP_GPU=${AOMP_GPU:-`$AOMP/bin/mygpu`}
PATH=$AOMP/bin:$PATH
CXX=clang++
FC=flang
FFLAGS="-O2  -target x86_64-pc-linux-gnu -fopenmp -fopenmp-targets=amdgcn-amd-amdhsa -Xopenmp-target=amdgcn-amd-amdhsa -march=$AOMP_GPU"
CXXFLAGS="-O2  -target x86_64-pc-linux-gnu -fopenmp -fopenmp-targets=amdgcn-amd-amdhsa -Xopenmp-target=amdgcn-amd-amdhsa -march=$AOMP_GPU"
OMP_TARGET_OFFLOAD=mandatory

export AOMP AOMP_GPU PATH CXX FC FFLAGS CXXFLAGS OMP_TARGET_OFFLOAD

cd $AOMP_REPOS_TEST/$AOMP_OVO_REPO_NAME
HALF_THREADS=$(( AOMP_JOB_THREADS/2 ))
echo "Using $HALF_THREADS threads for make."
export MAKEFLAGS=-j$HALF_THREADS

./ovo.sh run
./ovo.sh report --summary

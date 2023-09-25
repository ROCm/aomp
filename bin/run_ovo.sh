#!/bin/bash

# --- Start standard header to set AOMP environment variables ----
realpath=`realpath $0`
thisdir=`dirname $realpath`
export AOMP_USE_CCACHE=0

. $thisdir/aomp_common_vars
# --- end standard header ----

# Setup AOMP variables
AOMP=${AOMP:-/usr/lib/aomp}
FLANG=${FLANG:-flang}

# Use function to set and test AOMP_GPU
setaompgpu

AOMP_GPU=${AOMP_GPU:-`$AOMP/bin/mygpu`}
PATH=$AOMP/bin:$PATH
CXX=clang++
FC=$FLANG
FFLAGS="-O2  -target x86_64-pc-linux-gnu -fopenmp -fopenmp-targets=amdgcn-amd-amdhsa -Xopenmp-target=amdgcn-amd-amdhsa -march=$AOMP_GPU"
CXXFLAGS="-O2  -target x86_64-pc-linux-gnu -fopenmp -fopenmp-targets=amdgcn-amd-amdhsa -Xopenmp-target=amdgcn-amd-amdhsa -march=$AOMP_GPU"
OMP_TARGET_OFFLOAD=mandatory

export AOMP AOMP_GPU PATH CXX FC FFLAGS CXXFLAGS OMP_TARGET_OFFLOAD

cd $AOMP_REPOS_TEST/$AOMP_OVO_REPO_NAME
rm -f ovo.run.log*
HALF_THREADS=$(( AOMP_JOB_THREADS/2 ))
echo "Using $HALF_THREADS threads for make."
export MAKEFLAGS=-j$HALF_THREADS

./ovo.sh run
./ovo.sh report --summary

date=${BLOG_DATE:-`date '+%Y-%m-%d'`}
if [ "$1" == "log" ]; then
  if [ "$2" != "" ]; then
    prefix=$2
    log="$prefix/ovo.run.log.$date"
  else
    log="ovo.run.log.$date"
  fi
  echo "Log enabled: $log"
  ./ovo.sh report --failed 2>&1 | tee $log
  ./ovo.sh report --passed 2>&1 | tee -a $log
fi

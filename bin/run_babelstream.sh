#!/bin/bash

# run_babelstream.sh - runs babelstream in the $BABELSTREAM_BUILD dir.
# User can set RUN_OPTIONS to control which of the 4 variants to build and run
# The variants are:  omp-default, omp-fast, hip, and omp-cpu .
#
# This script allows overrides to the following environment variables:
#   BABELSTREAM_BUILD   - Build directory with results
#   BABELSTREAM_REPO    - Directory for the Babelstream repo
#   BABLESTREAM_REPEATS - The number of repititions
#   AOMP                - Compiler install directory
#   AOMPHIP             - HIP Compiler install directory, defaults to $AOMP
#   RUN_OPTIONS         - which of 4 variants to build and execute

# --- Start standard header to set AOMP environment variables ----
realpath=`realpath $0`
thisdir=`dirname $realpath`
. $thisdir/aomp_common_vars
# --- end standard header ----

# Set defaults for environment variables
AOMP=${AOMP:-/usr/lib/aomp}
AOMPHIP=${AOMPHIP:-$AOMP}

# note: clone_test.sh puts babelstream in $AOMP_REPOS_TEST/babelstream
BABELSTREAM_REPO=${BABELSTREAM_REPO:-$AOMP_REPOS_TEST/babelstream}
BABELSTREAM_BUILD=${BABELSTREAM_BUILD:-/tmp/$USER/babelstream}
BABELSTREAM_REPEATS=${BABELSTREAM_REPEATS:-10}

# Use function to set and test AOMP_GPU
setaompgpu
# skip hip if nvidida
if [ "${AOMP_GPU:0:3}" == "sm_" ] ; then
   TRIPLE="nvptx64-nvidia-cuda"
   RUN_OPTIONS=${RUN_OPTIONS:-"omp-default omp-fast"}
else
   TRIPLE="amdgcn-amd-amdhsa"
   RUN_OPTIONS=${RUN_OPTIONS:-"omp-default omp-fast hip"}
fi

omp_target_flags="-O3 -fopenmp -fopenmp-targets=$TRIPLE -Xopenmp-target=$TRIPLE -march=$AOMP_GPU -DOMP -DOMP_TARGET_GPU -Dsimd="
#  So this script runs with old comilers, we only use -fopenmp-target-fast
#  for LLVM 16 or higher
LLVM_VERSION=`$AOMP/bin/clang++ --version | grep version | cut -d" " -f 4 | cut -d"." -f1`
echo "LLVM VERSION IS $LLVM_VERSION"
if [[ $LLVM_VERSION -ge 16 ]] ; then
  omp_fast_flags="-fopenmp-gpu-threads-per-team=1024 -fopenmp-target-fast $omp_target_flags"
else
  if [[ $LLVM_VERSION -ge 15 ]] ; then
    omp_fast_flags="-fopenmp-target-ignore-env-vars -fopenmp-assume-no-thread-state -fopenmp-target-new-runtime $omp_target_flags"
  else
    if [[ $LLVM_VERSION -ge 14 ]] ; then
      omp_fast_flags="-fopenmp-target-new-runtime $omp_target_flags"
    else
      omp_fast_flags="$omp_target_flags"
    fi
  fi
fi
omp_cpu_flags="-O3 -fopenmp -DOMP"
hip_flags="-O3 --offload-arch=$AOMP_GPU -Wno-unused-result -DHIP -x hip"

omp_src="main.cpp OMPStream.cpp"
hip_src="main.cpp HIPStream.cpp"

gccver=`gcc --version | grep gcc | cut -d")" -f2 | cut -d"." -f1`
# ubunutu 18.04 gcc7 requires -stdlib=libc++ gcc9 is ok
if [ "$gccver" == " 7" ] ; then
   std="-std=c++20 -stdlib=libc++"
else
   std="-std=c++20"
fi

if [ ! -d $BABELSTREAM_REPO ]; then
  echo "ERROR: BabelStream not found in $BABELSTREAM_REPO"
  echo "       Consider running these commands:"
  echo
  echo "mkdir -p $AOMP_REPOS_TEST"
  echo "cd $AOMP_REPOS_TEST"
  echo "git clone https://github.com/UoB-HPC/babelstream"
  echo
  exit 1
fi
curdir=$PWD
mkdir -p $BABELSTREAM_BUILD
echo cd $BABELSTREAM_BUILD
cd $BABELSTREAM_BUILD
if [ "$1" != "nocopy" ] ; then
   cp $BABELSTREAM_REPO/src/main.cpp .
   cp $BABELSTREAM_REPO/src/Stream.h .
   cp $BABELSTREAM_REPO/src/omp/OMPStream.cpp .
   cp $BABELSTREAM_REPO/src/omp/OMPStream.cpp OMPStream.cpp.orig
   cp $BABELSTREAM_REPO/src/omp/OMPStream.h .
   cp $BABELSTREAM_REPO/src/hip/HIPStream.cpp .
   cp $BABELSTREAM_REPO/src/hip/HIPStream.h .
fi

echo RUN_OPTIONS: $RUN_OPTIONS
thisdate=`date`
echo >>results.txt
echo "=========> RUNDATE:  $thisdate" >>results.txt
COMPILER=`$AOMP/bin/llc --version  | grep version`
echo "=========> COMPILER: $COMPILER" >>results.txt
echo "=========> GPU:      $AOMP_GPU" >>results.txt
for option in $RUN_OPTIONS; do
  echo ; echo >>results.txt
  EXEC=stream-$option
  rm -f $EXEC
  if [ "$option" == "omp-default" ]; then
    compile_cmd="$AOMP/bin/clang++ $std $omp_target_flags $omp_src -o $EXEC"
  elif [ "$option" == "omp-fast" ]; then
    compile_cmd="$AOMP/bin/clang++ $std $omp_fast_flags   $omp_src -o $EXEC"
  elif [ "$option" == "omp-cpu" ]; then
    compile_cmd="$AOMP/bin/clang++ $std $omp_cpu_flags    $omp_src -o $EXEC"
  elif [ "$option" == "hip" ]; then
    if [ ! -f $AOMPHIP/bin/hipcc ] ; then 
      AOMPHIP="$AOMPHIP/.."
      if [ ! -f $AOMPHIP/bin/hipcc ] ; then 
        echo "ERROR: $AOMPHIP/bin/hipcc not found"
        exit 1
      fi
    fi
    compile_cmd="$AOMPHIP/bin/hipcc $std $hip_flags $hip_src -o $EXEC"
  else
    echo "ERROR: Option not recognized: $option."
    exit 1
  fi
  echo $compile_cmd | tee -a results.txt
  $compile_cmd
  if [ $? -ne 1 ]; then
     if [ -f $AOMP/bin/gpurun ] ; then
        echo $AOMP/bin/gpurun -s ./$EXEC -n $BABELSTREAM_REPEATS | tee -a results.txt
        $AOMP/bin/gpurun -s ./$EXEC -n $BABELSTREAM_REPEATS 2>&1 | tee -a results.txt
     else
        echo ./$EXEC -n $BABELSTREAM_REPEATS | tee -a results.txt
        ./$EXEC -n $BABELSTREAM_REPEATS 2>&1 | tee -a results.txt
     fi
  fi
done
cd $curdir
echo
echo "DONE. See results at end of file $BABELSTREAM_BUILD/results.txt"

#!/bin/bash

# run_babelstream.sh - runs babelstream in the $BABELSTREAM_BUILD dir.
# User can set RUN_OPTIONS to control what variants(omp_default, openmp, no_loop, hip) are selected.
# User can also set BABELSTREAM_BUILD,  BABELSTREAM_REPO, and AOMP env vars.
# The babelstream source OMPStream.cpp is patched to override number of teams
# and threads.

# --- Start standard header to set AOMP environment variables ----
realpath=`realpath $0`
thisdir=`dirname $realpath`
. $thisdir/aomp_common_vars
# --- end standard header ----

# Setup AOMP variables
AOMP=${AOMP:-/usr/lib/aomp}
AOMPHIP=${AOMPHIP:-$AOMP}
# clone_test.sh puts babelstream in $AOMP_REPOS_TEST/babelstream
BABELSTREAM_REPO=${BABELSTREAM_REPO:-$AOMP_REPOS_TEST/babelstream}
BABELSTREAM_BUILD=${BABELSTREAM_BUILD:-/tmp/$USER/babelstream}

# Use function to set and test AOMP_GPU
setaompgpu
if [ "${AOMP_GPU:0:3}" == "sm_" ] ; then
   TRIPLE="nvptx64-nvidia-cuda"
   RUN_OPTIONS=${RUN_OPTIONS:-"omp_default no_loop"}
else
   TRIPLE="amdgcn-amd-amdhsa"
   RUN_OPTIONS=${RUN_OPTIONS:-"omp_default no_loop hip"}
fi

omp_flags="-O3 -fopenmp -fopenmp-targets=$TRIPLE -Xopenmp-target=$TRIPLE -march=$AOMP_GPU -DOMP -DOMP_TARGET_GPU"
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
if [ "$1" != "nopatch" ] ; then 
   cp $BABELSTREAM_REPO/src/main.cpp .
   cp $BABELSTREAM_REPO/src/Stream.h .
   cp $BABELSTREAM_REPO/src/omp/OMPStream.cpp .
   cp $BABELSTREAM_REPO/src/omp/OMPStream.cpp OMPStream.cpp.orig
   cp $BABELSTREAM_REPO/src/omp/OMPStream.h .
   cp $BABELSTREAM_REPO/src/hip/HIPStream.cpp .
   cp $BABELSTREAM_REPO/src/hip/HIPStream.h .
   patch < $thisdir/patches/babelstream.patch
fi

echo RUN_OPTIONS: $RUN_OPTIONS
thisdate=`date`
echo >>results.txt
echo "=========> RUNDATE:  $thisdate" >>results.txt
COMPILER=`$AOMP/bin/llc --version  | grep version`
echo "=========> COMPILER: $COMPILER" >>results.txt
echo "=========> GPU:      $AOMP_GPU" >>results.txt
for option in $RUN_OPTIONS; do
  echo >>results.txt
  if [ "$option" == "openmp" ]; then
    EXEC=omp-stream
    rm -f $EXEC
    echo $AOMP/bin/clang++ $omp_flags $omp_src $std -o $EXEC
    $AOMP/bin/clang++ $omp_flags $omp_src $std -o $EXEC
    if [ $? -ne 1 ]; then
      ./$EXEC 2>&1 | tee -a results.txt
    fi
  elif [ "$option" == "omp_default" ]; then
    EXEC=omp-stream-default
    rm -f $EXEC
    echo $AOMP/bin/clang++ -D_DEFAULTS_NOSIMD $omp_flags $omp_src $std -o $EXEC
    $AOMP/bin/clang++ -D_DEFAULTS_NOSIMD $omp_flags $omp_src $std -o $EXEC
    if [ $? -ne 1 ]; then
      ./$EXEC 2>&1 | tee -a results.txt
    fi
  elif [ "$option" == "no_loop" ]; then
    # FAST_DOT requires merge of http://gerrit-git.amd.com/c/lightning/ec/llvm-project/+/705744
    $AOMP/bin/llvm-nm $AOMP/lib/libomptarget-new-amdgpu-gfx906.bc | grep kmpc_xteam_sum_d >/dev/null
    FAST_DOT_STATUS=$?
    # Check if new runtime is disabled via CCC_OVERRIDE_OPTIONS
    # Set new runtime status = 0 if new runtime has not been disabled
    echo "Checking CCC_OVERRIDE_OPTIONS"
    echo $CCC_OVERRIDE_OPTIONS | grep -ve '+-fno-openmp-target-new-runtime'
    NEW_RT_STATUS=$?
    FAST_DOT_STATUS=$((FAST_DOT_STATUS+NEW_RT_STATUS))
    [ $FAST_DOT_STATUS == 0 ] && omp_flags+=" -DOMP_TARGET_FAST_DOT"
    omp_flags+=" -fopenmp-target-ignore-env-vars"
    omp_flags+=" -fopenmp-assume-no-thread-state"
    omp_flags+=" -fopenmp-target-new-runtime"
    EXEC=omp-stream-no-loop
    rm -f $EXEC
    echo $AOMP/bin/clang++ -D_DEFAULTS_NOSIMD $omp_flags $omp_src $std -o $EXEC
    $AOMP/bin/clang++ -D_DEFAULTS_NOSIMD $omp_flags $omp_src $std -o $EXEC
    if [ $? -ne 1 ]; then
      ./$EXEC 2>&1 | tee -a results.txt
    fi
  elif [ "$option" == "omp_cpu" ]; then
    EXEC=omp-stream-cpu
    rm -f $EXEC
    echo $AOMP/bin/clang++ $omp_cpu_flags $omp_src $std -o $EXEC
    $AOMP/bin/clang++ $omp_cpu_flags $omp_src $std -o $EXEC
    if [ $? -ne 1 ]; then
      ./$EXEC 2>&1 | tee -a results.txt
    fi
  elif [ "$option" == "hip" ]; then
    EXEC=hip-stream
    rm -f $EXEC
    echo  $AOMP/bin/hipcc $hip_flags $hip_src $std -o $EXEC
    $AOMPHIP/bin/hipcc $hip_flags $hip_src $std -o $EXEC
    if [ $? -ne 1 ]; then
      ./$EXEC 2>&1 | tee -a results.txt
    fi
  else
    echo "ERROR: Option not recognized: $option."
    exit 1
  fi
done
cd $curdir
echo
echo "DONE. See results at end of file $BABELSTREAM_BUILD/results.txt"

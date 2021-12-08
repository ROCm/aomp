#!/bin/bash

# run_babelstream.sh - runs babelstream in the $BABELSTREAM_BUILD dir.
# User can set RUN_OPTIONS to control what variants(omp_default, openmp, hip) are selected.
# User can also set BABELSTREAM_BUILD,  BABELSTREAM_REPO, and AOMP env vars.
# The babelstream source OMPStream.cpp is patched to override number of teams
# and threads.

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
[ ! -L "$0" ] || thisdir=$(getdname `readlink "$0"`)
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

RUN_OPTIONS=${RUN_OPTIONS:-"omp_default openmp hip"}
omp_flags="-O3 -fopenmp -fopenmp-targets=amdgcn-amd-amdhsa -Xopenmp-target=amdgcn-amd-amdhsa -march=$AOMP_GPU -DOMP -DOMP_TARGET_GPU"
hip_flags="-O3 --offload-arch=$AOMP_GPU -DHIP -x hip"
omp_src="main.cpp OMPStream.cpp"
hip_src="main.cpp HIPStream.cpp"
std="-std=c++11"

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

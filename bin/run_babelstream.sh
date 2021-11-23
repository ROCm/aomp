#!/bin/bash

# run_babelstream.sh - runs babelstream in the $AOMP_REPOS_TEST dir.
# User can set RUN_OPTIONS to control what variants(openmp, hip) are selected.

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

# Use function to set and test AOMP_GPU
setaompgpu

RUN_OPTIONS=${RUN_OPTIONS:-"openmp hip"}
omp_flags="-O3 -fopenmp -fopenmp-targets=amdgcn-amd-amdhsa -Xopenmp-target=amdgcn-amd-amdhsa -march=$AOMP_GPU -DOMP -DOMP_TARGET_GPU"
hip_flags="-O3 --offload-arch=$AOMP_GPU -DHIP -x hip"
omp_src="main.cpp OMPStream.cpp"
hip_src="main.cpp HIPStream.cpp"
std="-std=c++11"

if [ -d $AOMP_REPOS_TEST/BabelStream ]; then
  cd $AOMP_REPOS_TEST/BabelStream
  rm -f results.txt
else
  echo "ERROR: BabelStream not found in $AOMP_REPOS_TEST."
  exit 1
fi

echo RUN_OPTIONS: $RUN_OPTIONS
for option in $RUN_OPTIONS; do
  if [ "$option" == "openmp" ]; then
    EXEC=omp-stream
    rm -f $EXEC
    echo $AOMP/bin/clang++ $omp_flags $omp_src $std -o $EXEC
    $AOMP/bin/clang++ $omp_flags $omp_src $std -o $EXEC
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
  echo >> results.txt
done

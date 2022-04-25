#!/bin/bash

# run_SU3Bench.sh - runs SU3Bench in the $AOMP_REPOS_TEST dir.
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
[ ! -L "$0" ] || thisdir=$(getdname `readlink -f "$0"`)
. $thisdir/aomp_common_vars
# --- end standard header ----
# Setup AOMP variables
AOMP=${AOMP:-/usr/lib/aomp}
AOMPHIP=${AOMPHIP:-$AOMP}

# Use function to set and test AOMP_GPU
setaompgpu

# Need thrust to use HIP
#RUN_OPTIONS=${RUN_OPTIONS:-"openmp hip"}
RUN_OPTIONS=${RUN_OPTIONS:-"openmp"}

#omp_flags="-O3 -fopenmp -fopenmp-targets=amdgcn-amd-amdhsa -Xopenmp-target=amdgcn-amd-amdhsa -march=$AOMP_GPU -DOMP -DOMP_TARGET_GPU"
#hip_flags="-O3 --offload-arch=$AOMP_GPU -DHIP -x hip"
#omp_src="main.cpp OMPStream.cpp"
#hip_src="main.cpp HIPStream.cpp"
#std="-std=c++11"

if [ -d $AOMP_REPOS_TEST/su3_bench ]; then
  cd $AOMP_REPOS_TEST/su3_bench
  rm -f results.txt
else
  echo "ERROR: su3bench found in $AOMP_REPOS_TEST."
  exit 1
fi

echo RUN_OPTIONS: $RUN_OPTIONS
for option in $RUN_OPTIONS; do
  if [ "$option" == "openmp" ]; then
    make -f Makefile.openmp clean
    export PATH=$AOMP/bin:$PATH
    make -f Makefile.openmp VENDOR=amd ARCH=MI200 all
    if [ $? -ne 1 ]; then
      ./bench_f32_openmp.exe 2>&1 | tee -a results.txt
      ./bench_f64_openmp.exe 2>&1 | tee -a results.txt
    fi
  elif [ "$option" == "hip" ]; then
    make -f Makefile.hip clean
    export PATH=$AOMP/bin:$PATH
    make -f Makefile.hip VENDOR=amd ARCH=MI200 all
    if [ $? -ne 1 ]; then
      ./bench_f32_hip.exe 2>&1 | tee -a results.txt
      ./bench_f64_hip.exe 2>&1 | tee -a results.txt
    fi
    cd ..
  else
    echo "ERROR: Option not recognized: $option."
    exit 1
  fi
  echo >> results.txt
done

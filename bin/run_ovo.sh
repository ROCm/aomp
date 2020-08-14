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
[ ! -L "$0" ] || thisdir=$(getdname `readlink "$0"`)
. $thisdir/aomp_common_vars
# --- end standard header ----

if [ "$AOMP" == "" ]; then
  echo "Error: Please set AOMP env variable and rerun."
  exit 1
fi

export AOMP_GPU=`$AOMP/bin/mygpu`
export PATH=$AOMP/bin:$PATH
export CXX=clang++
export FC=flang
export FFLAGS="-O2  -target x86_64-pc-linux-gnu -fopenmp -fopenmp-targets=amdgcn-amd-amdhsa -Xopenmp-target=amdgcn-amd-amdhsa -march=$AOMP_GPU"
export CXXFLAGS="-O2  -target x86_64-pc-linux-gnu -fopenmp -fopenmp-targets=amdgcn-amd-amdhsa -Xopenmp-target=amdgcn-amd-amdhsa -march=$AOMP_GPU"
export OMP_TARGET_OFFLOAD=mandatory

cd $AOMP_REPOS_TEST/$AOMP_OVO_REPO_NAME
HALF_THREADS=$(( NUM_THREADS/2 ))
echo "Using $HALF_THREADS threads for make."
export MAKEFLAGS=-j$HALF_THREADS

./ovo.sh run
./ovo.sh report --summary

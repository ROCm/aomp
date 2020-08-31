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

patchrepo $AOMP_REPOS_TEST/$AOMP_OMPTESTS_REPO_NAME


AOMP=${AOMP:-/usr/lib/aomp}
AOMP_GPU=`$AOMP/bin/mygpu`
DEVICE_ARCH=${DEVICE_ARCH:-$AOMP_GPU}
DEVICE_TARGET=${DEVICE_TARGET:-amdgcn-amd-amdhsa}


pushd $AOMP_REPOS_TEST/$AOMP_OMPTESTS_REPO_NAME
rm -f runtime-fails.txt
rm -f compile-fails.txt

# Move t-critical to avoid soft hang
if [ -d t-critical ]; then
  mv t-critical test-critical-fails
fi

log=$(date --iso-8601=minutes).log

env TARGET="-fopenmp-targets=$DEVICE_TARGET -fopenmp -Xopenmp-target=$DEVICE_TARGET -march=$DEVICE_ARCH" HOSTRTL=$AOMP/lib/libdevice TARGETRTL=$AOMP/lib GLOMPRTL=$AOMP/lib LLVMBIN=$AOMP/bin make -i 2>&1 | tee omptests_run_$log

# Get Results
compile_fails=0
runtime_fails=0
total_tests=$(ls | grep "^t\-*" | wc -l)
# Increment total tests, t-critical is skipped
((total_tests+=1))
if [ -e compile-fails.txt ]; then
  echo
  echo -----Compile Fails-----
  cat compile-fails.txt
  compile_fails=$(wc -l < compile-fails.txt)
fi
if [ -e runtime-fails.txt ]; then
  echo
  echo -----Runtime Fails-----
  cat runtime-fails.txt
  echo t-critical - soft hang
  runtime_fails=$(wc -l < runtime-fails.txt)
  # Increment fails, t-critical is skipped.
  ((runtime_fails+=1))
fi

successful_tests=$(( total_tests - compile_fails - runtime_fails ))
pass_rate=`bc -l <<< "scale=4; ($successful_tests/$total_tests) * 100" | sed "s/0//g"`
echo
echo ----- Results -----
echo Compile Fails: $compile_fails
echo Runtime Fails: $runtime_fails

echo Successful Tests: $successful_tests/$total_tests
echo Pass Rate: $pass_rate%
echo -------------------
echo
# Log Results
{
  echo
  echo ----- Results -----
  echo Compile Fails: $compile_fails
  echo Runtime Fails: $runtime_fails

  echo Successful Tests: $successful_tests/$total_tests
  echo Pass Rate: $pass_rate%
  echo -------------------
  echo
} >> omptests_run_$log
popd

removepatch $AOMP_REPOS_TEST/$AOMP_OMPTESTS_REPO_NAME

#!/bin/bash
# 
#  run_nekbone.sh: 
#

# --- Start standard header to set AOMP environment variables ----
realpath=`realpath $0`
thisdir=`dirname $realpath`
. $thisdir/aomp_common_vars
# --- end standard header ----

if [ "$1" == "rerun" ]; then
  cd $AOMP_REPOS_TEST/Nekbone
  cd test/nek_gpu1
  ulimit -s unlimited
  LIBOMPTARGET_KERNEL_TRACE=1 ./nekbone
  exit 0
fi

# Setup AOMP variables
AOMP=${AOMP:-/usr/lib/aomp}

# Use function to set and test AOMP_GPU
setaompgpu

cd $AOMP_REPOS_TEST/Nekbone
cd test/nek_gpu1
make -f makefile.aomp clean
ulimit -s unlimited
PATH=$AOMP/bin/:$PATH make -f makefile.aomp
LIBOMPTARGET_KERNEL_TRACE=1 ./nekbone 2>&1 | tee nek.log
grep -s Exitting nek.log
exit $?

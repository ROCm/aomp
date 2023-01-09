#!/bin/bash
# 
#  run_nekbone.sh: 
#

# --- Start standard header to set AOMP environment variables ----
realpath=`realpath $0`
thisdir=`dirname $realpath`
. $thisdir/aomp_common_vars
# --- end standard header ----

export CCC_OVERRIDE_OPTIONS="+-O1" 
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
rm -f make-fail.txt failing-tests.txt passing-tests.txt
make -f makefile.aomp clean
ret=$?
if [ $ret -ne 0 ]; then
  echo "nekbone" >> make-fail.txt
  exit 1
fi
ulimit -s unlimited
PATH=$AOMP/bin/:$PATH make -f makefile.aomp
VERBOSE=${VERBOSE:-"1"}
if [ $VERBOSE -eq 0 ]; then
  LIBOMPTARGET_KERNEL_TRACE=1 ./nekbone 2>&1 | tee nek.log > /dev/null
else
  LIBOMPTARGET_KERNEL_TRACE=1 ./nekbone 2>&1 | tee nek.log
fi
grep -s Exitting nek.log
ret=$?
if [ $ret -ne 0 ]; then
  echo "nekbone" >> failing-tests.txt
else
  echo "nekbone" >> passing-tests.txt
  if [ $VERBOSE -eq 0 ]; then
    tail -7 nek.log
  fi
fi
echo "running Nekbone at -O1, AAPointer issue"
exit $ret

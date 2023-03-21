#!/bin/bash

function clean(){
  rm -f test_acquire_release
  rm -f test_allocate_directive
  rm -f test_concurrent_map
  rm -f test_declare_variant
  rm -f test_imperfect_loop
  rm -f test_implicit_declare_target
  rm -f test_non_rectangular
  rm -f test_requires_unified_shared_memory
  rm -f test_use_device_addr
  rm -f *.o
  rm -f *.ERR
  rm -f LLNL.run.log*
}

clean

AOMP=${AOMP:-/usr/lib/aomp}
AOMP_GPU=${AOMP_GPU:-`$AOMP/bin/mygpu`}
export AOMP AOMP_GPU
echo AOMP_GPU = $AOMP_GPU
echo AOMP = $AOMP
date=`date '+%Y-%m-%d'`
if [ "$1" == "log" ]; then
  if [ "$2" != "" ]; then
    prefix=$2
    log="$prefix/LLNL.run.log.$date"
  else
    log="LLNL.run.log.$date"
  fi
  echo "Log enabled: $log"
 ./test.py $AOMP $AOMP_GPU 2>&1 | tee $log
else
 ./test.py $AOMP $AOMP_GPU
fi

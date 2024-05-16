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

if [ -e /usr/sbin/lspci ]; then
 lspci_loc=/usr/sbin/lspci
else
  if [ -e /sbin/lspci ]; then
    lspci_loc=/sbin/lspci
  else
    lspci_loc=/usr/bin/lspci
  fi
fi
echo $lspci_loc
$lspci_loc 2>&1 | grep -s VMware

if [ $? -ne 0 ] ; then
  grep -s test_requires_unified_shared_memory.cxx test_list
  if [ $? -ne 0 ] ; then
    echo test_requires_unified_shared_memory.cxx >> test_list
    echo test_requires_unified_shared_memory.cxx added
  fi
fi
AOMP=${AOMP:-/usr/lib/aomp}
AOMP_GPU=${AOMP_GPU:-`$AOMP/bin/mygpu`}
export AOMP AOMP_GPU
echo AOMP_GPU = $AOMP_GPU
echo AOMP = $AOMP
date=${BLOG_DATE:-`date '+%Y-%m-%d'`}
if [ "$1" == "log" ]; then
  if [ "$2" != "" ]; then
    prefix=$2
    log="$prefix/LLNL.run.log.$date"
  else
    log="LLNL.run.log.$date"
  fi
  echo "Log enabled: $log"
 timeout 120 ./test.py $AOMP $AOMP_GPU 2>&1 | tee $log
else
 timeout 120 ./test.py $AOMP $AOMP_GPU
fi

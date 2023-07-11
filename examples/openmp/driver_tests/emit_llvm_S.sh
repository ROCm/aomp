#!/bin/bash
#
#  show-offload-types.sh:  Show GPU offloading, host offloading and no offloading
#
# find AOMP if not set in an environment variable 
AOMP=${AOMP:-$HOME/rocm/aomp}
[[ -f $AOMP/bin/clang ]] || AOMP=/usr/lib/aomp
if [ ! -f $AOMP/bin/clang ] ; then
  echo please build or install AOMP
  exit 1
fi
tdir=/tmp/s$$
mkdir -p $tdir
tmpcfile=$tdir/test.c
/bin/cat >$tmpcfile <<"EOF"
#include <stdio.h>
#include <omp.h>
int main() {
printf("HOST:   initial_device:%d  default_device:%d   num_devices:%d\n",
  omp_get_initial_device(),omp_get_default_device(),omp_get_num_devices());
#pragma omp target teams
#pragma omp parallel
if((omp_get_thread_num()==0) && (omp_get_team_num()==0))
  printf("TARGET: device_num:%d  num_devices:%d get_initial:%d  is_initial:%d \n        threads:%d  teams:%d\n",
  omp_get_device_num(),omp_get_num_devices(),omp_get_initial_device(),omp_is_initial_device(),
  omp_get_num_threads(),omp_get_num_teams());
}
EOF
echo
echo "======== $tmpcfile ========"
cat $tmpcfile
echo; echo "========== COMPILES ==========="

gpu=`$AOMP/bin/offload-arch`
if [  -z $gpu ]; then
  echo offload-arch fails, fixing
  gpu=`rocminfo |grep amdgcn-amd-amdhsa--gfx | head -1|awk '{print $2}'|sed s/amdgcn-amd-amdhsa--//`
  echo "$gpu"
fi

targetoptions="-fopenmp-targets=amdgcn-amd-amdhsa -Xopenmp-target=amdgcn-amd-amdhsa -march=$gpu --no-offload-new-driver"
RC=0

cmd="$AOMP/bin/clang $tmpcfile -fopenmp -emit-llvm -S"
echo "$cmd" ; $cmd
file test.ll

cmd="$AOMP/bin/clang $tmpcfile -fopenmp -emit-llvm -c"
echo "$cmd" ; $cmd
file test.bc

cmd="$AOMP/bin/clang $tmpcfile -fopenmp -S"
echo "$cmd" ; $cmd
file test.s

cmd="$AOMP/bin/clang $tmpcfile -fopenmp -emit-llvm -S $targetoptions"
echo "$cmd" ; $cmd
file test.ll
grep "target triple = \"amdgcn-amd-amdhsa\"" test.ll

if [ $? -eq 0 ]; then 
	echo passed
else
   echo failed "-emit-llvm -S"
   RC=1
fi

cmd="$AOMP/bin/clang $tmpcfile -fopenmp -emit-llvm -c $targetoptions"
echo "$cmd" ; $cmd
file test.bc

cmd="$AOMP/bin/clang $tmpcfile -fopenmp -S $targetoptions"
echo "$cmd" ; $cmd
file test.s
grep ".amdgcn_target \"amdgcn-amd-amdhsa--gfx" test.s
 
if [ $? -eq 0 ]; then 
	echo passed
else
   echo failed "-S"
   RC=1
fi

# cleanup
rm -rf $tdir
echo

exit $RC

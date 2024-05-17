#!/bin/bash
#
#  show-offload-types.sh:  Show GPU offloading, host offloading and no offloading
#
if [ ! -f $LLVM_INSTALL_DIR/bin/clang ] ; then
  echo please build or install an LLVM compiler and correctly set LLVM_INSTALL_DIR
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

if [  -z $LLVM_GPU_ARCH ]; then
  echo amdgpu-arch fails, fixing
  LLVM_GPU_ARCH=`rocminfo |grep amdgcn-amd-amdhsa--gfx | head -1|awk '{print $2}'|sed s/amdgcn-amd-amdhsa--//`
  echo "$LLVM_GPU_ARCH"
fi

targetoptions="--offload-arch=$LLVM_GPU_ARCH"
RC=0

cmd="$LLVM_INSTALL_DIR/bin/clang $tmpcfile -fopenmp -emit-llvm -S"
echo "$cmd" ; $cmd
file test.ll

cmd="$LLVM_INSTALL_DIR/bin/clang $tmpcfile -fopenmp -emit-llvm -c"
echo "$cmd" ; $cmd
file test.bc

cmd="$LLVM_INSTALL_DIR/bin/clang $tmpcfile -fopenmp -S"
echo "$cmd" ; $cmd
file test.s

cmd="$LLVM_INSTALL_DIR/bin/clang $tmpcfile -fopenmp -emit-llvm -c $targetoptions"
echo "$cmd" ; $cmd
file test.bc

# cleanup
rm -rf $tdir
echo

exit $RC

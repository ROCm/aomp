#!/bin/bash
#
#  show-offload-types.sh:  Show GPU offloading, host offloading and no offloading
#
# find AOMP if not set in an environment variable 
AOMP=${AOMP:-$HOME/rocm/aomp}
[[ -f $AOMP/bin/clang ]] ||  AOMP=/usr/lib/aomp
if [ ! -f $AOMP/bin/clang ] ; then 
  echo please build or install AOMP
  exit 1
fi
tmpcfile=/tmp/test.c
tdir=/tmp
/bin/cat >$tmpcfile <<"EOF"
#include <stdio.h>
#include <omp.h>
int main() { 
printf("HOST: omp_get_initial_device :%d num_devices:%d\n", omp_get_initial_device(),omp_get_num_devices()); 
#pragma omp target teams
#pragma omp parallel
if((omp_get_thread_num()==0) && (omp_get_team_num()==0))
  printf("TARGET: omp_get_device_num:%d num_devices:%d threads:%d teams:%d\n", 
  omp_get_device_num(),omp_get_num_devices(),omp_get_num_threads(),omp_get_num_teams()); 
}
EOF
echo
echo "==== $tmpcfile ===="
cat $tmpcfile
echo "==== COMPILES ===="

gpu=`$AOMP/bin/offload-arch`
cmd="$AOMP/bin/clang -fopenmp -o $tdir/gpu-offload   $tmpcfile --offload-arch=$gpu"
echo "$cmd" ; $cmd
cmd="$AOMP/bin/clang -fopenmp -o $tdir/offload-xnack $tmpcfile --offload-arch=$gpu:xnack+"
echo "$cmd" ; $cmd
cmd="$AOMP/bin/clang -fopenmp -o $tdir/host-offload  $tmpcfile --offload-arch=znver1"
echo "$cmd" ; $cmd
cmd="$AOMP/bin/clang -fopenmp -o $tdir/no-offload    $tmpcfile"
echo "$cmd" ; $cmd

echo ; echo  ==== gpu-offload ====
$tdir/gpu-offload
echo ; echo  ==== host-offload ====
$tdir/host-offload
echo ; echo  ==== no-offload ====
$tdir/no-offload
echo ; echo  ==== gpu-offload a missing requirement ======
$tdir/offload-xnack
echo ; echo  ==== gpu-offload with offload disabled appears as if no-offload ====
echo export OMP_TARGET_OFFLOAD=DISABLED
export OMP_TARGET_OFFLOAD=DISABLED
$tdir/gpu-offload
echo ; echo  ==== host-offload with offload disabled appears as if no-offload ====
$tdir/host-offload
echo ; echo  ==== host-offload with offload mandatory ======
echo export OMP_TARGET_OFFLOAD=MANDATORY
export OMP_TARGET_OFFLOAD=MANDATORY
$tdir/host-offload
echo ; echo  ==== gpu-offload with offload mandatory and missing requirement ======
$tdir/offload-xnack
echo

# cleanup
echo "====== CLEANUP ===="
cmd="rm $tmpcfile $tdir/gpu-offload $tdir/host-offload $tdir/no-offload $tdir/offload-xnack"
echo "$cmd" ; $cmd
echo 


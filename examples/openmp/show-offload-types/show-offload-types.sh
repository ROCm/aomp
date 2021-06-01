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
tmpcfile=/tmp/test.c
tdir=/tmp
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

[[ -f $tdir/no-offload ]] && rm $tdir/no-offload
[[ -f $tdir/host-offload ]] && rm $tdir/host-offload
[[ -f $tdir/gpu-offload ]] && rm $tdir/gpu-offload
[[ -f $tdir/qualed-offload ]] && rm $tdir/qualed-offload

cmd="$AOMP/bin/clang $tmpcfile -fopenmp -o $tdir/no-offload     "
echo "$cmd" ; $cmd

cmd="$AOMP/bin/clang $tmpcfile -fopenmp -o $tdir/host-offload   --offload-arch=x86"
echo "$cmd" ; $cmd

cmd="$AOMP/bin/clang $tmpcfile -fopenmp -o $tdir/gpu-offload    --offload-arch=$gpu"
echo "$cmd" ; $cmd

# This does not work for nvidia devices 
if [[ "${gpu##*sm_}" == "${gpu}" ]] ; then
  cmd="$AOMP/bin/clang $tmpcfile -fopenmp -o $tdir/qualed-offload --offload-arch=$gpu:xnack+"
  echo "$cmd" ; $cmd
fi


echo; echo "========== DEFAULT EXECUTION ==========="
if [[ -f $tdir/no-offload ]] ; then
  echo ; echo  ==== no-offload ==== ; $tdir/no-offload
fi

if [[ -f $tdir/host-offload ]] ; then
  echo ; echo  X=== host-offload ==== ; $tdir/host-offload
fi

if [[ -f $tdir/gpu-offload ]] ; then
 echo ; echo  ==== gpu-offload ==== ; $tdir/gpu-offload
fi
 
if [[ -f $tdir/qualed-offload ]] ; then
  echo ; echo  ==== qualed-offload with no qualified devices ==== ; $tdir/qualed-offload
fi

echo; echo "========== EXECUTE DISABLED ==========="
echo export OMP_TARGET_OFFLOAD=DISABLED
export OMP_TARGET_OFFLOAD=DISABLED

if [[ -f $tdir/no-offload ]] ; then
  echo ; echo  ==== no-offload with offload disabled === ; $tdir/no-offload
fi

if [[ -f $tdir/host-offload ]] ; then
  echo ; echo  ==== host-offload with offload disabled appears as if no-offload ==== ; $tdir/host-offload
fi

if [[ -f $tdir/gpu-offload ]] ; then
  echo ; echo  ==== gpu-offload with offload disabled appears as if no-offload ==== ; $tdir/gpu-offload
fi

if [[ -f $tdir/qualed-offload ]] ; then
  echo ; echo  ==== qualed-offload with offload disabled and no qualified devices === ; $tdir/qualed-offload
fi

echo; echo "========== EXECUTE MANDATORY ==========="
echo export OMP_TARGET_OFFLOAD=MANDATORY
export OMP_TARGET_OFFLOAD=MANDATORY

if [[ -f $tdir/no-offload ]] ; then
  echo ; echo  X=== no-offload with offload mandatory === SHOULD THIS FAIL ?? ; $tdir/no-offload
fi

if [[ -f $tdir/host-offload ]] ; then
  echo ; echo  X=== host-offload with offload mandatory === ; $tdir/host-offload
fi

if [[ -f $tdir/gpu-offload ]] ; then
  echo ; echo  ==== gpu-offload with offload mandatory === ; $tdir/gpu-offload
fi

if [[ -f $tdir/qualed-offload ]] ; then
  echo ; echo  X=== qualed-offload with offload mandatory and no qualified devices === ; $tdir/qualed-offload
fi

# cleanup
rm $tmpcfile
[[ -f $tidir/no-offload ]] && rm $tdir/no-offload
[[ -f $tdir/host-offload ]] && rm $tdir/host-offload
[[ -f $tdir/gpu-offload ]] && rm $tdir/gpu-offload
[[ -f $tdir/qualed-offload ]] && rm $tdir/qualed-offload

echo

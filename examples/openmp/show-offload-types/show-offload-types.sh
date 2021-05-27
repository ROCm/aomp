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
echo; echo "========== COMPILES ==========="

gpu=`$AOMP/bin/offload-arch`

[[ -f $tdir/gpu-offload ]] && rm $tdir/gpu-offload
[[ -f $tdir/offload-xnack ]] && rm $tdir/offload-xnack
[[ -f $tdir/host-offload ]] && rm $tdir/host-offload
[[ -f $tdir/no-offload ]] && rm $tdir/no-offload

cmd="$AOMP/bin/clang -fopenmp -o $tdir/gpu-offload   $tmpcfile --offload-arch=$gpu"
echo; echo "$cmd" ; $cmd

if [[ "${gpu##*sm_}" == "${gpu}" ]] ; then 
   echo AMD
  cmd="$AOMP/bin/clang -fopenmp -o $tdir/offload-xnack $tmpcfile --offload-arch=$gpu:xnack+"
  echo; echo "$cmd" ; $cmd
fi

cmd="$AOMP/bin/clang -fopenmp -o $tdir/host-offload  $tmpcfile --offload-arch=znver1"
echo; echo "$cmd" ; $cmd

cmd="$AOMP/bin/clang -fopenmp -o $tdir/no-offload    $tmpcfile"
echo; echo "$cmd" ; $cmd



echo; echo "========== EXECUTE==========="
if [[ -f $tdir/gpu-offload ]] ; then 
 echo ; echo  ==== gpu-offload ==== ; $tdir/gpu-offload
fi
 
if [[ -f $tdir/host-offload ]] ; then 
  echo ; echo  ==== host-offload ==== ; $tdir/host-offload 
fi

if [[ -f $tdir/no-offload ]] ; then 
  echo ; echo  ==== no-offload ==== ; $tdir/no-offload
fi

if [[ -f $tdir/offload-xnack ]] ; then 
  echo ; echo  ==== gpu-offload a missing requirement ==== ; $tdir/offload-xnack
fi

echo; echo export OMP_TARGET_OFFLOAD=DISABLED
export OMP_TARGET_OFFLOAD=DISABLED

if [[ -f $tdir/gpu-offload ]] ; then 
  echo ; echo  ==== gpu-offload with offload disabled appears as if no-offload ==== ; $tdir/gpu-offload
fi

if [[ -f $tdir/host-offload ]] ; then 
  echo ; echo  ==== host-offload with offload disabled appears as if no-offload ==== ; $tdir/host-offload
fi

echo; echo export OMP_TARGET_OFFLOAD=MANDATORY
export OMP_TARGET_OFFLOAD=MANDATORY

if [[ -f $tdir/host-offload ]] ; then 
  echo ; echo  ==== host-offload with offload mandatory === ; $tdir/host-offload
fi

if [[ -f $tdir/offload-xnack ]] ; then 
  echo ; echo  ==== gpu-offload with offload mandatory and missing requirement === ; $tdir/offload-xnack
fi

# cleanup
rm $tmpcfile 
[[ -f $tdir/gpu-offload ]] && rm $tdir/gpu-offload 
[[ -f $tdir/host-offload ]] && rm $tdir/host-offload 
[[ -f $tidir/no-offload ]] && rm $tdir/no-offload
[[ -f $tdir/offload-xnack ]] && rm $tdir/offload-xnack

echo

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

cmd="$LLVM_INSTALL_DIR/bin/clang $tmpcfile -fopenmp -o $tdir/no-offload     "
echo "$cmd" ; $cmd

### cmd="$LLVM_INSTALL_DIR/bin/clang $tmpcfile -fopenmp -o $tdir/host-offload   --offload-arch=x86_64"
### echo "$cmd" ; $cmd

cmd="$LLVM_INSTALL_DIR/bin/clang $tmpcfile -fopenmp -o $tdir/gpu-offload --offload-arch=$LLVM_GPU_ARCH"
echo "$cmd" ; $cmd

# This does not work for nvidia devices 
if [[ "${LLVM_GPU_ARCH##*sm_}" == "${LLVM_GPU_ARCH}" ]] ; then
  if [ "$LLVM_GPU_ARCH" == "gfx90a" ] || [ "$LLVM_GPU_ARCH" == "gfx940" ] ; then
    cmd="env HSA_XNACK=1 $LLVM_INSTALL_DIR/bin/clang $tmpcfile -fopenmp -o $tdir/qualed-offload --offload-arch=$LLVM_GPU_ARCH:xnack+"
    echo "$cmd"
    $cmd
  fi
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
  if [ "$LLVM_GPU_ARCH" == "gfx90a" ] || [ "$LLVM_GPU_ARCH" == "gfx940" ] ; then
    echo ; echo  ==== qualed-offload with no qualified devices ==== ; HSA_XNACK=1 $tdir/qualed-offload
  fi
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
  if [ "$LLVM_GPU_ARCH" == "gfx90a" ] || [ "$LLVM_GPU_ARCH" == "gfx940" ] ; then
    echo ; echo  ==== qualed-offload with offload disabled and no qualified devices === ; HSA_XNACK=1 $tdir/qualed-offload
  fi
fi

echo; echo "========== EXECUTE MANDATORY ==========="
echo export OMP_TARGET_OFFLOAD=MANDATORY
export OMP_TARGET_OFFLOAD=MANDATORY

if [[ -f $tdir/no-offload ]] ; then
  echo ; echo  X=== no-offload with offload mandatory === ; $tdir/no-offload
fi

if [[ -f $tdir/host-offload ]] ; then
  echo ; echo  X=== host-offload with offload mandatory === ; $tdir/host-offload
fi

if [[ -f $tdir/gpu-offload ]] ; then
  echo ; echo  X=== gpu-offload with offload mandatory === ; $tdir/gpu-offload
fi

if [[ -f $tdir/qualed-offload ]] ; then
  if [ "$LLVM_GPU_ARCH" == "gfx90a" ] || [ "$LLVM_GPU_ARCH" == "gfx940" ] ; then
  echo ; echo  X=== qualed-offload with offload mandatory and no qualified devices === ; HSA_XNACK=1 $tdir/qualed-offload
  fi
fi

# cleanup
rm -rf $tdir
echo

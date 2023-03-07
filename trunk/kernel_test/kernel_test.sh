#!/bin/bash
#
# driver_test.sh : create two heterogeneous objects and link to main c code. 
#                  to demonstrate offloading driver. 
#
# Three environment variables are used by this script:
#    TRUNK:  Location of LLVM trunk installation, default is $HOME/rocm/trunk
#    OARCH:  Offload architecture, default is sm_70 
#    USE_FLANG: Use flang-new, default is NO because flang offload driver still broken
#
# Example: To use clang to avoid flang-new lane function test , run:
#   USE_FLANG=NO ./driver_test.sh

# Set environment variable defaults here:
TRUNK=${TRUNK:-$HOME/rocm/trunk}
USE_FLANG=${USE_FLANG:-YES}
OFFLOAD=${OFFLOAD:-MANDATORY} # Also use disabled

if [ ! -f $TRUNK/bin/amdgpu-arch ] ; then
  OARCH=${OARCH:-sm_70}
  echo "WARNING, no amdgpu-arch utility in $TRUNK to get current offload-arch, using $OARCH"
else
  amdarch=`$TRUNK/bin/amdgpu-arch | head -n 1`
  OARCH=${OARCH:-$amdarch}
fi

_llvm_bin_dir=$TRUNK/bin

#extra_args="-v -save-temps"
extra_args="-save-temps"

main_c_binary="main_in_c"
[ -f $main_c_binary ] && rm $main_c_binary
compile_main_cmd="$_llvm_bin_dir/clang -save-temps -fopenmp --offload-arch=$OARCH  main.c -o $main_c_binary"
echo
echo $compile_main_cmd
$compile_main_cmd
echo OMP_TARGET_OFFLOAD=$OFFLOAD ./$main_c_binary
OMP_TARGET_OFFLOAD=$OFFLOAD ./$main_c_binary
rc=$?
echo "RETURN CODE IS: $rc"
echo "CONVERTING temp bc files to ll.  See files host_c.ll, device_c.ll"
#   Eventually host_f.ll device_f.ll
$TRUNK/bin/llvm-dis main-host-x86_64-unknown-linux-gnu.bc -o host_c.ll
$TRUNK/bin/llvm-dis main-openmp-amdgcn-amd-amdhsa-gfx908.tmp.bc -o device_c.ll
echo "-----------------------------------------------------"
echo "===> HOST function defs and calls in host_c.ll"
echo
grep "define\|call" host_c.ll | grep -v requires | tee host_calls.txt
echo
echo "-----------------------------------------------------"
echo "===> DEVICE function defs and calls in device_c.ll"
echo
grep "define\|call" device_c.ll | tee device_calls.txt
echo 
echo "-----------------------------------------------------"
echo


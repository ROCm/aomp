#!/bin/bash
#
# kernel_test.sh : Compare c and fortran versions of write_index
#
# Three environment variables are used by this script:
#    TRUNK:  Location of LLVM trunk installation, default is $HOME/rocm/trunk
#    OFFLOAD: MANDATORY OR DISABLED
#    OARCH:  Offload architecture, sm_70, gfx908, etc
#
# Set environment variable defaults here:
TRUNK=${TRUNK:-$HOME/rocm/trunk}
OFFLOAD=${OFFLOAD:-DISABLED}

if [ ! -f $TRUNK/bin/amdgpu-arch ] ; then
  OARCH=${OARCH:-sm_70}
  echo "WARNING, no amdgpu-arch utility in $TRUNK to get current offload-arch, using $OARCH"
else
  amdarch=`$TRUNK/bin/amdgpu-arch | head -n 1`
  OARCH=${OARCH:-$amdarch}
fi

_llvm_bin_dir=$TRUNK/bin

#extra_args="-v -fno-integrated-as -save-temps"
flang_extra_args="-fno-integrated-as -save-temps"
clang_extra_args="-save-temps"

tmpc="tmpc"
rm -rf $tmpc ; mkdir -p $tmpc ; cd $tmpc
[ -f main_in_c ] && rm main_in_c
compile_main_cmd="$_llvm_bin_dir/clang $clang_extra_args -fopenmp --offload-arch=$OARCH  ../main.c -o main_in_c"
echo
echo $compile_main_cmd
$compile_main_cmd
echo OMP_TARGET_OFFLOAD=$OFFLOAD ./main_in_c
OMP_TARGET_OFFLOAD=$OFFLOAD ./main_in_c
rc=$?
echo "C RETURN CODE IS: $rc"

echo "CONVERTING temp bc files to ll.  See files host_c.ll, device_c.ll"
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

cd ..
tmpf="tmpf"
rm -rf $tmpf ; mkdir -p $tmpf ; cd $tmpf
[ -f main_in_f ] && rm main_in_f
compile_main_f_cmd="$_llvm_bin_dir/flang-new $flang_extra_args -flang-experimental-exec -fopenmp --offload-arch=$OARCH ../main.f95 -o main_in_f"
#compile_main_f_cmd="$_llvm_bin_dir/flang-new $flang_extra_args -fopenmp ../main.f95 -o main_in_f"
echo
echo $compile_main_f_cmd
$compile_main_f_cmd
if [ -f main_in_f ] ; then 
   echo
   echo OMP_TARGET_OFFLOAD=$OFFLOAD ./main_in_f
   OMP_TARGET_OFFLOAD=$OFFLOAD ./main_in_f
   rc=$?
   echo "FORTRAN RETURN CODE IS: $rc"
else
   echo "COMPILE FAILED, SKIPPING EXECUTION"
fi
echo "CONVERTING temp bc files to ll"
if [ -f main.bc ] ; then  
   $TRUNK/bin/llvm-dis main.bc -o host_no_offload.ll
fi
if [ -f main-host-x86_64-unknown-linux-gnu.bc ] ; then 
   $TRUNK/bin/llvm-dis main-host-x86_64-unknown-linux-gnu.bc -o host_f.ll
   echo "-----------------------------------------------------"
   echo "===> HOST function defs and calls in host_f.ll"
   echo
   grep "define\|call" host_f.ll | grep -v requires | tee host_calls.txt
fi
if [ -f main-openmp-amdgcn-amd-amdhsa-gfx908.tmp.bc ] ; then 
   $TRUNK/bin/llvm-dis main-openmp-amdgcn-amd-amdhsa-gfx908.tmp.bc -o device_f.ll
   echo "-----------------------------------------------------"
   echo "===> DEVICE function defs and calls in device_f.ll"
   echo
   grep "define\|call" device_f.ll
fi
echo "-----------------------------------------------------"
cd ..

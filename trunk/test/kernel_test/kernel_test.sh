#!/bin/bash
#
# kernel_test.sh : Compare c and fortran versions of write_index
#
# Three environment variables are used by this script:
#    TRUNK:  Location of LLVM trunk installation, default is $HOME/rocm/trunk
#    OFFLOAD: MANDATORY OR DISABLED
#    OARCH:  Offload architecture, sm_70, gfx908 , etc
#
# Set environment variable defaults here:
TRUNK=${TRUNK:-$HOME/rocm/trunk}
OFFLOAD=${OFFLOAD:-MANDATORY}

_offload_arch=`$TRUNK/bin/amdgpu-arch 2>/dev/null | head -n 1`
if [ -z "$_offload_arch" ] ; then
  _offload_arch=`$TRUNK/bin/nvptx-arch 2>/dev/null | head -n 1`
fi
if [ -z "$_offload_arch" ] ; then
   echo error no arch found
   exit 1
fi
OARCH=${OARCH:-$_offload_arch}
if [ ${OARCH:0:3} == "gfx" ] ; then
   TRIPLE="amdgcn-amd-amdhsa"
else
   TRIPLE="nvptx64-nvidia-cuda"
fi

_llvm_bin_dir=$TRUNK/bin

flang_extra_args="-v -save-temps"
clang_extra_args="-v -save-temps"

tmpc="tmpc"
echo
echo "++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++"
echo "++++++++++  START c demo, in directory $tmpc  ++++++++++++++"
echo "++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++"
rm -rf $tmpc ; mkdir -p $tmpc ; cd $tmpc
echo cd $tmpc
[ -f main_in_c ] && rm main_in_c
compile_main_cmd="$_llvm_bin_dir/clang $clang_extra_args -fopenmp --offload-arch=$OARCH  ../main.c -o main_in_c"
echo
echo "$compile_main_cmd 2>stderr_save_temps"
$compile_main_cmd 2>stderr_save_temps
echo "LIBOMPTARGET_DEBUG=1 OMP_TARGET_OFFLOAD=$OFFLOAD ./main_in_c 2>debug.out"
LIBOMPTARGET_DEBUG=1 OMP_TARGET_OFFLOAD=$OFFLOAD ./main_in_c 2>debug.out
rc=$?
echo "C RETURN CODE IS: $rc"

echo "CONVERTING temp bc files to ll.  See files host_c.ll, device_c.ll"
$TRUNK/bin/llvm-dis main-host-x86_64-unknown-linux-gnu.bc -o host_c.ll
$TRUNK/bin/llvm-dis main-openmp-$TRIPLE-$OARCH.bc -o device_c.ll
echo "-----------------------------------------------------"
echo "===> HOST function defs and calls in tmpc/host_c.ll"
grep "define\|call" host_c.ll | grep -v requires | grep -v nocallback | grep -v "@llvm\." | grep -v ompx_no_call | tee host_calls.txt
echo "-----------------------------------------------------"
echo "===> DEVICE function defs and calls in tmpc/device_c.ll"
grep "define\|call" device_c.ll | grep -v nocallback | grep -v "@llvm\." | grep -v ompx_no_call | tee device_calls.txt
echo "-----------------------------------------------------"
echo
tmpf="tmpf"
echo "++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++"
echo "+++++++  END c demo, begin FORTRAN demo in dir $tmpf +++++++"
echo "++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++"
cd ..
rm -rf $tmpf ; mkdir -p $tmpf ; cd $tmpf
echo cd $tmpf
[ -f main_in_f ] && rm main_in_f
#_experimental_arg="-flang-experimental-exec"
_experimental_arg=""
compile_main_f_cmd="$_llvm_bin_dir/flang-new $flang_extra_args $_experimental_arg -fopenmp --offload-arch=$OARCH ../main.f95 -o main_in_f"
echo
echo "$compile_main_f_cmd 2>stderr_save_temps"
$compile_main_f_cmd 2>stderr_save_temps
if [ -f main_in_f ] ; then 
   echo
   echo "LIBOMPTARGET_DEBUG=1 OMP_TARGET_OFFLOAD=$OFFLOAD ./main_in_f 2>debug.out"
   LIBOMPTARGET_DEBUG=1 OMP_TARGET_OFFLOAD=$OFFLOAD ./main_in_f 2>debug.out
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
   echo "===> HOST function defs and calls in tmpf/host_f.ll"
   grep "define\|call" host_f.ll | grep -v requires | grep -v nocallback | grep -v "@llvm\." | grep -v "@llvm\." | grep -v ompx_no_call | tee host_calls.txt
fi
if [ -f main-openmp-$TRIPLE-$OARCH.bc ] ; then 
   $TRUNK/bin/llvm-dis main-openmp-$TRIPLE-$OARCH.bc -o device_f.ll
   echo "-----------------------------------------------------"
   echo "===> DEVICE function defs and calls in tmpf/device_f.ll"
   grep "define\|call" device_f.ll | grep -v nocallback | grep -v "@llvm\." | grep -v ompx_no_call
fi
echo "-----------------------------------------------------"
cd ..

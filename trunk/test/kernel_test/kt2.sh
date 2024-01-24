#!/bin/bash
#
# kernel_test.sh : Compare c and fortran versions of write_index
#
# Three environment variables are used by this script:
#    TRUNK:  Location of LLVM trunk installation, default is $HOME/rocm/trunk
#    OFFLOAD: MANDATORY OR DISABLED
#    OARCH:  Offload architecture, sm_70, gfx908, etc
# 
# kt2.sh has the following differences from kernel_test.sh 
#    - remove -save-temps compile option
#    - use directories tmpc2 and tmpf2 instead of tmpc and tmpf
#    - Remove bc disassembly because no bc without --save-temps
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

flang_extra_args="-v"
clang_extra_args="-v"

tmpc="tmpc2"
echo
echo "++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++"
echo "++++++++++  START c demo, in directory $tmpc  ++++++++++++++"
echo "++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++"
rm -rf $tmpc ; mkdir -p $tmpc ; cd $tmpc
echo cd $tmpc
[ -f main_in_c ] && rm main_in_c
compile_main_cmd="$_llvm_bin_dir/clang $clang_extra_args -fopenmp --offload-arch=$OARCH  ../main.c -o main_in_c"
echo
echo "$compile_main_cmd 2>stderr_nosave"
$compile_main_cmd 2>>stderr_nosave
echo "LIBOMPTARGET_DEBUG=1 OMP_TARGET_OFFLOAD=$OFFLOAD ./main_in_c 2>debug.out"
LIBOMPTARGET_DEBUG=1 OMP_TARGET_OFFLOAD=$OFFLOAD ./main_in_c 2>debug.out
rc=$?
echo "C RETURN CODE IS: $rc"
echo
tmpf="tmpf2"
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
echo "$compile_main_f_cmd 2>stderr_nosave"
$compile_main_f_cmd 2>>stderr_nosave
if [ -f main_in_f ] ; then 
   echo
   echo "LIBOMPTARGET_DEBUG=1 OMP_TARGET_OFFLOAD=$OFFLOAD ./main_in_f 2>debug.out"
   LIBOMPTARGET_DEBUG=1 OMP_TARGET_OFFLOAD=$OFFLOAD ./main_in_f 2>debug.out
   rc=$?
   echo "FORTRAN RETURN CODE IS: $rc"
else
   echo "COMPILE FAILED, SKIPPING EXECUTION"
fi
cd ..

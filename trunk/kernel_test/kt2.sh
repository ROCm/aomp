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
#    use directories tmpc2 and tmpf2
#    remove -save-temps but use -v to see toolchain commands in stderr_nosave
#    remove -fno-integrated-as
#    turn on LIBOMPTARGET_DEBUG to save trace in debug.out
#    set OMP_TARGET_OFFLOAD to MANDATORY 
#    Remove host and device IR disasssembly because we dont have --save-temps
#    Add output from -ccc-print-phase 
#
# Set environment variable defaults here:
TRUNK=${TRUNK:-$HOME/rocm/trunk}
OFFLOAD=${OFFLOAD:-MANDATORY}

if [ ! -f $TRUNK/bin/amdgpu-arch ] ; then
  OARCH=${OARCH:-sm_70}
  echo "WARNING, no amdgpu-arch utility in $TRUNK to get current offload-arch, using $OARCH"
else
  amdarch=`$TRUNK/bin/amdgpu-arch | head -n 1`
  OARCH=${OARCH:-$amdarch}
fi

_llvm_bin_dir=$TRUNK/bin

#extra_args="-v -fno-integrated-as -save-temps"
flang_extra_args="-v"
clang_extra_args="-v"

tmpc="tmpc2"
rm -rf $tmpc ; mkdir -p $tmpc ; cd $tmpc
echo cd $tmpc
[ -f main_in_c ] && rm main_in_c
compile_main_cmd="$_llvm_bin_dir/clang $clang_extra_args -fopenmp --offload-arch=$OARCH  ../main.c -o main_in_c"
echo
echo $compile_main_cmd
echo $compile_main_cmd >stderr_nosave.c
$compile_main_cmd 2>>stderr_nosave.c
$compile_main_cmd -ccc-print-phases 2>>stderr_nosave.c
echo "LIBOMPTARGET_DEBUG=1 OMP_TARGET_OFFLOAD=$OFFLOAD ./main_in_c 2>debug.out"
LIBOMPTARGET_DEBUG=1 OMP_TARGET_OFFLOAD=$OFFLOAD ./main_in_c 2>debug.out
rc=$?
echo "C RETURN CODE IS: $rc"
echo "++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++"
echo "+++++++++++++  END c demo, begin FORTRAN demo  +++++++++++++"
echo "++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++"
cd ..
tmpf="tmpf2"
rm -rf $tmpf ; mkdir -p $tmpf ; cd $tmpf
echo cd $tmpf
[ -f main_in_f ] && rm main_in_f
compile_main_f_cmd="$_llvm_bin_dir/flang-new $flang_extra_args -flang-experimental-exec -fopenmp --offload-arch=$OARCH ../main.f95 -o main_in_f"
echo
echo $compile_main_f_cmd
echo $compile_main_f_cmd >stderr_nosave.f
$compile_main_f_cmd 2>>stderr_nosave.f
$compile_main_f_cmd -ccc-print-phases  2>>stderr_nosave.f
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

#!/bin/bash
#
# parallel_test.sh : Compare c and fortran versions of write_index
#
# Three environment variables are used by this script:
#    TRUNK:  Location of LLVM trunk installation, default is $HOME/rocm/trunk
#    OFFLOAD: MANDATORY OR DISABLED, default is MANDATORY
#    OARCH:  Offload architecture, sm_70, gfx908, etc
#    FLANG: binary name for flang compiler, default is flang-new
#
# To test this script with AOMP legacy flang compiler
#    export TRUNK=$AOMP (where AOMP is installed)
#    export FLANG=flang-legacy
#
# Set environment variable defaults here:
TRUNK=${TRUNK:-$HOME/rocm/trunk}
OFFLOAD=${OFFLOAD:-MANDATORY}
_realpath=`realpath $0`
_thisdir=`dirname $_realpath`
_curdir=$PWD
FLANG=${FLANG:-flang-new}
_llvm_bin_dir=$TRUNK/bin
if [ ! -f $_llvm_bin_dir/$FLANG ] ; then
  echo
  echo "ERROR: Compiler executable $_llvm_bin_dir/$FLANG is missing."
  echo "       Consider setting env var FLANG to flang"
  echo
  exit 1
fi

if [ ! -f $TRUNK/bin/amdgpu-arch ] ; then
  OARCH=${OARCH:-sm_70}
  echo "WARNING, no amdgpu-arch utility in $TRUNK to get current offload-arch, using $OARCH"
else
  amdarch=`$TRUNK/bin/amdgpu-arch | head -n 1`
  OARCH=${OARCH:-$amdarch}
fi

flang_extra_args="-v -save-temps"
clang_extra_args="-v -save-temps"

tmpc="tmpc"
echo
echo mkdir -p $tmpc
echo cd $tmpc
rm -rf $tmpc ; mkdir -p $tmpc ; cd $tmpc
echo "++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++"
echo "++++++++++  START c demo, in directory $tmpc  ++++++++++++++"
echo "++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++"
[ -f main_in_c ] && rm main_in_c
compile_main_cmd="$_llvm_bin_dir/clang $clang_extra_args -fopenmp --offload-arch=$OARCH  $_thisdir/main.c -o main_in_c"
echo
echo "$compile_main_cmd 2>$tmpc/stderr_save_temps"
$compile_main_cmd 2>stderr_save_temps
echo "LIBOMPTARGET_DEBUG=1 OMP_TARGET_OFFLOAD=$OFFLOAD ./main_in_c 2>$tmpc/debug.out"
LIBOMPTARGET_DEBUG=1 OMP_TARGET_OFFLOAD=$OFFLOAD ./main_in_c 2>debug.out
rc=$?
echo "C RETURN CODE IS: $rc"

echo "CONVERTING temp bc files to ll.  See files host_c.ll, device_c.ll"
$TRUNK/bin/llvm-dis main-host-x86_64-unknown-linux-gnu.bc -o host_c.ll
_bcfile="main-openmp-amdgcn-amd-amdhsa-$OARCH.tmp.bc"
if [ -f $_bcfile ] ; then
   $TRUNK/bin/llvm-dis $_bcfile -o device_c.ll
   echo "-----------------------------------------------------"
   echo "===> HOST function defs and calls in tmpc/host_c.ll"
   grep "define\|call" host_c.ll | grep -v requires | grep -v nocallback | tee host_calls.txt
   echo "-----------------------------------------------------"
   echo "===> DEVICE function defs and calls in tmpc/device_c.ll"
   grep "define\|call" device_c.ll | grep -v nocallback | tee device_calls.txt
   echo "-----------------------------------------------------"
fi
echo
tmpf="tmpf"
echo
echo "++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++"
echo "+++++  END c demo, begin FORTRAN demo in dir $tmpf  ++++++++"
echo "++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++"
cd $_curdir
echo mkdir -p $tmpf
rm -rf $tmpf ; mkdir -p $tmpf ; cd $tmpf
echo cd $tmpf
[ -f main_in_f ] && rm main_in_f
compile_main_f_cmd="$_llvm_bin_dir/$FLANG $flang_extra_args -fopenmp --offload-arch=$OARCH $_thisdir/main.f95 -o main_in_f"
echo
echo "$compile_main_f_cmd 2>$tmpf/stderr_save_temps"
$compile_main_f_cmd 2>stderr_save_temps
if [ -f main_in_f ] ; then 
   echo
   echo "LIBOMPTARGET_DEBUG=1 OMP_TARGET_OFFLOAD=$OFFLOAD ./main_in_f 2>$tmpf/debug.out"
   LIBOMPTARGET_DEBUG=1 OMP_TARGET_OFFLOAD=$OFFLOAD ./main_in_f 2>debug.out
   _script_rc=$?
   echo "FORTRAN RETURN CODE IS: $_script_rc"
   if LIBOMPTARGET_INFO=-1 ./main_in_f 2>&1 | grep  "Tripcount: 10000"
   then
      echo "Loop tripcount calculated correctly"
   else
      echo "Loop tripcount mismatch. Expected loop tripcount 10000."
      _script_rc=1
   fi
else
   echo "COMPILE FAILED, SKIPPING EXECUTION"
   _script_rc=1
fi
if [ -f main.bc ] ; then  
   echo "CONVERTING temp main.bc files to main.ll"
   $TRUNK/bin/llvm-dis main.bc -o main.ll
fi
echo 
if [ -f main-host-x86_64-unknown-linux-gnu.bc ] ; then 
   $TRUNK/bin/llvm-dis main-host-x86_64-unknown-linux-gnu.bc -o host_f.ll
   echo "-----------------------------------------------------"
   echo "===> HOST function defs and calls in tmpf/host_f.ll"
   grep "define\|call" host_f.ll | grep -v requires | grep -v nocallback | tee host_calls.txt
fi
_bcfile="main-openmp-amdgcn-amd-amdhsa-$OARCH.tmp.bc"
if [ -f $_bcfile ] ; then
   $TRUNK/bin/llvm-dis $_bcfile -o device_f.ll
   echo "-----------------------------------------------------"
   echo "===> DEVICE function defs and calls in tmpf/device_f.ll"
   grep "define\|call" device_f.ll | grep -v nocallback
fi
echo "-----------------------------------------------------"
cd $_curdir
exit $_script_rc

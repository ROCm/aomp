#!/bin/bash
#
# pt2.sh : Compare c and fortran versions of a test of reduction do
#
# Three environment variables are used by this script:
#    TRUNK:  Location of LLVM trunk installation, default is $HOME/rocm/trunk
#    OFFLOAD: MANDATORY OR DISABLED, default is MANDATORY
#    OARCH:  Offload architecture, sm_70, gfx908, etc
#    FLANG: binary name for flang compiler, default is flang-new
#
# To test this script with the AOMP legacy flang compiler, set these vars
#    export TRUNK=$AOMP (where AOMP is installed)
#    export FLANG=flang-legacy
#
# pt2.sh is identical to reduction_test.sh except pt2.sh
#    does not use -save-temps because sometimes -save-temps changes behavior or fails
#    outputs to directories tmpc2 and tmpf2 instead of tmpc and tmpf
#    toolchain commands from -v are saved to file stderr_nosave instead of stderr_save_temps
#    has no llvm disassembly of .bc files since save-temps is off
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

flang_extra_args="-v"
clang_extra_args="-v"

tmpc="tmpc2"
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
echo "$compile_main_cmd 2>$tmpc/stderr_nosave"
$compile_main_cmd 2>>stderr_nosave
echo "LIBOMPTARGET_DEBUG=1 OMP_TARGET_OFFLOAD=$OFFLOAD ./main_in_c 2>$tmpc/debug.out"
LIBOMPTARGET_DEBUG=1 OMP_TARGET_OFFLOAD=$OFFLOAD ./main_in_c 2>debug.out
rc=$?
echo "C RETURN CODE IS: $rc"
echo
tmpf="tmpf2"
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
echo "$compile_main_f_cmd 2>$tmpf/stderr_nosave"
$compile_main_f_cmd 2>>stderr_nosave
if [ -f main_in_f ] ; then
   echo
   echo "LIBOMPTARGET_DEBUG=1 OMP_TARGET_OFFLOAD=$OFFLOAD ./main_in_f 2>$tmpf/debug.out"
   LIBOMPTARGET_DEBUG=1 OMP_TARGET_OFFLOAD=$OFFLOAD ./main_in_f 2>debug.out
   _script_rc=$?
   echo "FORTRAN RETURN CODE IS: $_script_rc"
else
   echo "COMPILE FAILED, SKIPPING EXECUTION"
   _script_rc=1
fi
cd $_curdir
exit $_script_rc

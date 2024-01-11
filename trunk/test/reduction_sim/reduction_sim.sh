#!/bin/bash
#
# reduction_sim.sh : Compare c and fortran versions of reduction_sim 
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
  echo
  echo "WARNING, no amdgpu-arch utility in $TRUNK to get current offload-arch, using $OARCH"
  echo
else
  amdarch=`$TRUNK/bin/amdgpu-arch | head -n 1`
  OARCH=${OARCH:-$amdarch}
fi

_gpurun="$TRUNK/bin/gpurun"
[ ! -f $_gpurun ] && _gpurun="/opt/rocm/llvm/bin/gpurun"
[ ! -f $_gpurun ] && _gpurun=""

ROCMINFO_BINARY="$TRUNK/bin/rocminfo"
[ ! -f $ROCMINFO_BINARY ] && ROCMINFO_BINARY="/opt/rocm/bin/rocminfo"
[ -f $ROCMINFO_BINARY ] && export ROCMINFO_BINARY

flang_extra_args="-v -save-temps -O3 -fopenmp --offload-arch=$OARCH"
clang_extra_args="-O3 -v -save-temps -fopenmp --offload-arch=$OARCH"

_s=$_thisdir
omp_fsrc="$_s/reduction_sim.f95"
omp_csrc="$_s/reduction_sim.c"


tmpc="tmpc"
echo
echo mkdir -p $tmpc
echo cd $tmpc
rm -rf $tmpc ; mkdir -p $tmpc ; cd $tmpc
echo "++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++"
echo "++++++++++  START c demo, in directory $tmpc   ++++++++++++"
echo "++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++"
echo
echo $_llvm_bin_dir/clang -c -o helper_fns_cpu.o $_s/helper_fns_cpu.c
$_llvm_bin_dir/clang -c -o helper_fns_cpu.o $_s/helper_fns_cpu.c
echo 
[ -f main_in_c ] && rm main_in_c
compile_main_c_cmd="$_llvm_bin_dir/clang $clang_extra_args $omp_csrc helper_fns_cpu.o -o main_in_c"
echo
echo "$compile_main_c_cmd 2>$tmpc/stderr_save_temps"
$compile_main_c_cmd 2>stderr_save_temps

if [ ! -f main_in_c ] ; then
   echo "ERROR: COMPILE FAILED see $tmpc/stderr_save_temps"
   echo
   cat stderr_save_temps
   echo 
   echo "CMD:$compile_main_c_cmd"
   echo
   cd $_curdir
   exit 1
fi
echo "OMP_TARGET_OFFLOAD=$OFFLOAD $_gpurun ./main_in_c 2>$tmpc/debug.out"
LIBOMPTARGET_DEBUG=1 OMP_TARGET_OFFLOAD=$OFFLOAD $_gpurun ./main_in_c 2>debug.out | tee c_results
rc=$?
echo "c EXECUTION RETURN CODE IS: $rc"

echo "CONVERTING temp bc files to ll.  See files $tmpc/host_c.ll, $tmpc/device_c.ll"
$TRUNK/bin/llvm-dis reduction_sim-host-x86_64-unknown-linux-gnu.bc -o host_c.ll
$TRUNK/bin/llvm-dis reduction_sim-openmp-amdgcn-amd-amdhsa-gfx908.tmp.bc -o device_c.ll
echo "===> finding HOST function defs and calls in tmpc/host_c.ll , see $tmpc/host_calls.txt"
grep "define\|call" host_c.ll | grep -v requires | grep -v nocallback > host_calls.txt
echo "===> finding DEVICE function defs and calls in tmpc/device_c.ll, see $tmpc/device_calls.txt"
grep "define\|call" device_c.ll | grep -v nocallback | grep -v lifetime >  device_calls.txt

tmpf="tmpf"
echo
echo "++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++"
echo "+++++    END c demo, begin FORTRAN demo in dir $tmpf  ++++++"
echo "++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++"

cd $_curdir
echo mkdir -p $tmpf
rm -rf $tmpf ; mkdir -p $tmpf ; cd $tmpf
echo cd $tmpf
echo
echo $_llvm_bin_dir/clang -c -o helper_fns_cpu.o $_s/helper_fns_cpu.c
$_llvm_bin_dir/clang -c -o helper_fns_cpu.o $_s/helper_fns_cpu.c
echo 
[ -f main_in_f ] && rm main_in_f
compile_main_f_cmd="$_llvm_bin_dir/$FLANG $flang_extra_args $omp_fsrc helper_fns_cpu.o -o main_in_f"

echo
echo "$compile_main_f_cmd 2>$tmpf/stderr_save_temps"
$compile_main_f_cmd 2>stderr_save_temps
if [ -f main_in_f ] ; then 
   echo
   echo "LIBOMPTARGET_DEBUG=1 OMP_TARGET_OFFLOAD=$OFFLOAD ./main_in_f 2>$tmpf/debug.out"
   LIBOMPTARGET_DEBUG=1 OMP_TARGET_OFFLOAD=$OFFLOAD ./main_in_f 2>debug.out 
   _script_rc=$?
   echo "FORTRAN RETURN CODE IS: $_script_rc"
   [ $_script_rc != 0 ] && echo see $tmpf/debug.out 
else
   echo "COMPILE FAILED, SKIPPING EXECUTION , see $tmpf/stderr_save_temps"
   cat stderr_save_temps
   echo 
   echo "CMD:$compile_main_f_cmd"
   echo
   _script_rc=1
fi
echo "CONVERTING temp bc files to ll.  See files $tmpf/host_f.ll, $tmpf/device_f.ll"
$TRUNK/bin/llvm-dis reduction_sim-host-x86_64-unknown-linux-gnu.bc -o host_f.ll
$TRUNK/bin/llvm-dis reduction_sim-openmp-amdgcn-amd-amdhsa-gfx908.tmp.bc -o device_f.ll
echo "===> finding HOST function defs and calls in tmpf/host_f.ll , see $tmpf/host_calls.txt"
grep "define\|call" host_f.ll | grep -v requires | grep -v nocallback > host_calls.txt
echo "===> finding DEVICE function defs and calls in tmpf/device_f.ll, see $tmpf/device_calls.txt"
grep "define\|call" device_f.ll | grep -v nocallback | grep -v lifetime >  device_calls.txt

cd $_curdir
exit $_script_rc

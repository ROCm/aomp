#!/bin/bash
#
# stream_no_teams.sh : Compare c and fortran versions of stream (no teams)
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

_gpurun="$TRUNK/bin/gpurun"
[ ! -f $_gpurun ] && _gpurun="/opt/rocm/llvm/bin/gpurun"
[ ! -f $_gpurun ] && _gpurun=""

ROCMINFO_BINARY="$TRUNK/bin/rocminfo"
[ ! -f $ROCMINFO_BINARY ] && ROCMINFO_BINARY="/opt/rocm/bin/rocminfo"
[ -f $ROCMINFO_BINARY ] && export ROCMINFO_BINARY

flang_extra_args="-v -save-temps -O3 -fopenmp --offload-arch=$OARCH -DVERSION_STRING=4.0 -DUSE_OPENMPTARGET=1"
clang_extra_args="-v -save-temps -std=c++11 -O3 -fopenmp --offload-arch=$OARCH  -DOMP -DOMP_TARGET_GPU -Dsimd= "

_s=$_thisdir/src
omp_fsrc="$_s/BabelStreamTypes.F90 $_s/ArrayStream.F90 $_s/OpenMPTargetStream.F90 $_s/main.F90"

tmpc="tmpc"
echo
echo mkdir -p $tmpc
echo cd $tmpc
rm -rf $tmpc ; mkdir -p $tmpc ; cd $tmpc
echo "++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++"
echo "++++++++++  START c++ demo, in directory $tmpc   ++++++++++++"
echo "++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++"
[ -f main_in_c ] && rm main_in_c
compile_main_cmd="$_llvm_bin_dir/clang++ $clang_extra_args $_thisdir/src/main.cpp $_thisdir/src/OMPStream.cpp  -o main_in_c"
echo
echo "$compile_main_cmd 2>$tmpc/stderr_save_temps"
$compile_main_cmd 2>stderr_save_temps

if [ ! -f main_in_c ] ; then 
   echo "ERROR: COMPILE FAILED see $tmpc/stderr_save_temps"
   echo 
   exit 1
fi
echo "OMP_TARGET_OFFLOAD=$OFFLOAD $_gpurun ./main_in_c -n 10 2>$tmpc/debug.out"
OMP_TARGET_OFFLOAD=$OFFLOAD $_gpurun ./main_in_c -n 10 2>debug.out | tee c_results
rc=$?
echo "EXECUTION RETURN CODE IS: $rc"

echo "CONVERTING temp bc files to ll.  See files $tmpc/host_c.ll, $tmpc/device_c.ll"
$TRUNK/bin/llvm-dis OMPStream-host-x86_64-unknown-linux-gnu.bc -o host_c.ll
$TRUNK/bin/llvm-dis OMPStream-openmp-amdgcn-amd-amdhsa-gfx908.tmp.bc -o device_c.ll
echo "===> finding HOST function defs and calls in tmpc/host_c.ll , see $tmpc/host_calls.txt"
grep "define\|call" host_c.ll | grep -v requires | grep -v nocallback > host_calls.txt
echo "===> finding DEVICE function defs and calls in tmpc/device_c.ll, see $tmpc/device_calls.txt"
grep "define\|call" device_c.ll | grep -v nocallback | grep -v lifetime >  device_calls.txt
tmpf="tmpf"
echo
echo "++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++"
echo "+++++  END c++ demo, begin FORTRAN demo in dir $tmpf  ++++++"
echo "++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++"

cd $_curdir
echo mkdir -p $tmpf
rm -rf $tmpf ; mkdir -p $tmpf ; cd $tmpf
echo cd $tmpf
[ -f main_in_f ] && rm main_in_f
compile_main_f_cmd="$_llvm_bin_dir/$FLANG $flang_extra_args $omp_fsrc -o main_in_f"
echo
echo "$compile_main_f_cmd 2>$tmpf/stderr_save_temps"
$compile_main_f_cmd 2>stderr_save_temps
if [ -f main_in_f ] ; then 
   echo
   echo "OMP_TARGET_OFFLOAD=$OFFLOAD $_gpurun ./main_in_f -n 10 2>$tmpf/debug.out"
   OMP_TARGET_OFFLOAD=$OFFLOAD $_gpurun ./main_in_f -n 10 2>debug.out | tee f_results
   _script_rc=$?
   echo "FORTRAN RETURN CODE IS: $_script_rc"
else
   echo "COMPILE FAILED, SKIPPING EXECUTION , see $tmpf/stderr_save_temps"
   _script_rc=1
fi
if [ -f main.bc ] ; then  
   echo "CONVERTING temp main.bc files to main.ll"
   $TRUNK/bin/llvm-dis main.bc -o main.ll
fi
echo 
if [ -f main-host-x86_64-unknown-linux-gnu.bc ] ; then 
   $TRUNK/bin/llvm-dis main-host-x86_64-unknown-linux-gnu.bc -o host_f.ll
   echo "===> finding HOST function defs and calls in tmpf/host_f.ll, see $tmpf/host_calls.txt"
   grep "define\|call" host_f.ll | grep -v requires | grep -v nocallback > host_calls.txt
fi
if [ -f main-openmp-amdgcn-amd-amdhsa-gfx908.tmp.bc ] ; then 
   $TRUNK/bin/llvm-dis main-openmp-amdgcn-amd-amdhsa-gfx908.tmp.bc -o device_f.ll
   echo "===> finding DEVICE function defs and calls in tmpf/device_f.ll, see $tmpf/device_calls.txt"
   grep "define\|call" device_f.ll | grep -v nocallback >device_calls.txt
fi
cd $_curdir
exit $_script_rc

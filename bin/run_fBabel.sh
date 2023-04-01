#!/bin/bash

# run_babelstream.sh - runs babelstream in the $FABELSTREAM_BUILD dir.
# User can set RUN_OPTIONS to control which of the 4 variants to build and run
# The variants are:  omp-default, omp-fast, and omp-cpu .
#
# This script allows overrides to the following environment variables:
#   FABELSTREAM_BUILD   - Build directory with results
#   FABELSTREAM_REPO    - Directory for the Babelstream repo
#   BABLESTREAM_REPEATS - The number of repititions
#   AOMP                - Compiler install directory
#   RUN_OPTIONS         - which of 4 variants to build and execute

# --- Start standard header to set AOMP environment variables ----
realpath=`realpath $0`
thisdir=`dirname $realpath`
. $thisdir/aomp_common_vars
# --- end standard header ----

# Set defaults for environment variables
AOMP=${AOMP:-/usr/lib/aomp}

# note: clone_test.sh puts babelstream in $AOMP_REPOS_TEST/babelstream
FABELSTREAM_REPO=${FABELSTREAM_REPO:-$AOMP_REPOS_TEST/fortran.babelstream}
FABELSTREAM_BUILD=${FABELSTREAM_BUILD:-/tmp/$USER/fbabelstream}
FABELSTREAM_REPEATS=${FABELSTREAM_REPEATS:-10}

# Clean validation files
rm -f "$FABELSTREAM_REPO"/passing-tests.txt "$FABELSTREAM_REPO"/failing-tests.txt "$FABELSTREAM_REPO"/make-fail.txt

# Use function to set and test AOMP_GPU
setaompgpu
# skip hip if nvidida
if [ "${AOMP_GPU:0:3}" == "sm_" ] ; then
   TRIPLE="nvptx64-nvidia-cuda"
   RUN_OPTIONS=${RUN_OPTIONS:-"omp-default omp-fast"}
else
   TRIPLE="amdgcn-amd-amdhsa"
   RUN_OPTIONS=${RUN_OPTIONS:-"omp-default omp-fast"}
fi

omp_target_flags="-O3 -fopenmp -fopenmp-targets=$TRIPLE -Xopenmp-target=$TRIPLE -march=$AOMP_GPU -DOMP -DUSE_OPENMPTARGET=1 -DVERSION_STRING=4.0"
#  So this script runs with old comilers, we only use -fopenmp-target-fast
#  for LLVM 16 or higher
LLVM_VERSION=`$AOMP/bin/flang --version | grep version | cut -d" " -f 3 | cut -d"." -f1`
if [ "$LLVM_VERSION" == "version" ]  ; then
  # If 3rd  arg is version, must be vendor cmppiler, so get version from 4th field.
  LLVM_VERSION=`$AOMP/bin/flang --version | grep version | cut -d" " -f 4 | cut -d"." -f1`
  special_aso_flags="-O3 -fopenmp-target-fast"
  #special_aso_flags="-fopenmp-gpu-threads-per-team=256 -fopenmp-target-fast"
else
  # temp hack to detect trunk (not vendor compiler) (found version in 3 args)"
  special_aso_flags=""
fi
echo "LLVM VERSION IS $LLVM_VERSION"
if [[ $LLVM_VERSION -ge 16 ]] ; then
  omp_fast_flags="$special_aso_flags $omp_target_flags"
else
  if [[ $LLVM_VERSION -ge 15 ]] ; then
    omp_fast_flags="-fopenmp-target-ignore-env-vars -fopenmp-assume-no-thread-state -fopenmp-target-new-runtime $omp_target_flags"
  else
    if [[ $LLVM_VERSION -ge 14 ]] ; then
      omp_fast_flags="-fopenmp-target-new-runtime $omp_target_flags"
    else
      omp_fast_flags="$omp_target_flags"
    fi
  fi
fi
omp_cpu_flags="-O3 -fopenmp -DOMP"

omp_src="BabelStreamTypes.F90 ArrayStream.F90 OpenMPTargetStream.F90 main.F90"

gccver=`gcc --version | grep gcc | cut -d")" -f2 | cut -d"." -f1`
std=""

if [ ! -d $FABELSTREAM_REPO ]; then
  echo "ERROR: BabelStream not found in $FABELSTREAM_REPO"
  echo "       Consider running these commands:"
  echo
  echo "mkdir -p $AOMP_REPOS_TEST"
  echo "cd $AOMP_REPOS_TEST"
  echo "git clone -b develop https://github.com/UoB-HPC/babelstream fortran.babelstream"
  echo
  exit 1
fi
curdir=$PWD
mkdir -p $FABELSTREAM_BUILD
echo cd $FABELSTREAM_BUILD
cd $FABELSTREAM_BUILD
if [ "$1" != "nocopy" ] ; then
   cp $FABELSTREAM_REPO/src/fortran/main.F90 .
   cp $FABELSTREAM_REPO/src/fortran/ArrayStream.F90 .
   cp $FABELSTREAM_REPO/src/fortran/BabelStreamTypes.F90 .
   cp $FABELSTREAM_REPO/src/fortran/OpenMPTargetStream.F90 .
   cp $FABELSTREAM_REPO/src/fortran/OpenMPTargetLoopStream.F90 .
fi

echo RUN_OPTIONS: $RUN_OPTIONS
thisdate=`date`
echo >>results.txt
echo "=========> RUNDATE:  $thisdate" >>results.txt
COMPILER=`$AOMP/bin/llc --version  | grep version`
echo "=========> COMPILER: $COMPILER" >>results.txt
echo "=========> GPU:      $AOMP_GPU" >>results.txt
compile_error=0
runtime_error=0

for option in $RUN_OPTIONS; do
  echo ; echo >>results.txt
  EXEC=stream-$option
  rm -f $EXEC
  if [ "$option" == "omp-default" ]; then
    compile_cmd="$AOMP/bin/flang $std $omp_target_flags $omp_src -o $EXEC"
  elif [ "$option" == "omp-fast" ]; then
    compile_cmd="$AOMP/bin/flang $std $omp_fast_flags   $omp_src -o $EXEC"
  elif [ "$option" == "omp-cpu" ]; then
    compile_cmd="$AOMP/bin/flang $std $omp_cpu_flags    $omp_src -o $EXEC"
  else
    echo "ERROR: Option not recognized: $option."
    exit 1
  fi
  echo $compile_cmd | tee -a results.txt
  $compile_cmd
  if [ $? -ne 0 ]; then
    compile_error=1
    echo "babelstream" >> "$FABELSTREAM_REPO"/make-fail.txt
    break
  else
    set -o pipefail
    if [ -f $AOMP/bin/gpurun ] ; then
      echo $AOMP/bin/gpurun -s ./$EXEC -n $FABELSTREAM_REPEATS | tee -a results.txt
      $AOMP/bin/gpurun -s ./$EXEC -n $FABELSTREAM_REPEATS 2>&1 | tee -a results.txt
      if [ $? -ne 0 ]; then
        runtime_error=1
      fi
    else
      echo ./$EXEC -n $FABELSTREAM_REPEATS | tee -a results.txt
      ./$EXEC -n $FABELSTREAM_REPEATS 2>&1 | tee -a results.txt
      if [ $? -ne 0 ]; then
        runtime_error=1
      fi
    fi
  fi
done

# Check for errors
if [ $compile_error -ne 0 ]; then
  echo "babelstream" >> "$FABELSTREAM_REPO"/make-fail.txt
elif [ $runtime_error -ne 0 ]; then
  echo "babelstream" >> "$FABELSTREAM_REPO"/failing-tests.txt
else
  echo "babelstream" >> "$FABELSTREAM_REPO"/passing-tests.txt
fi

cd $curdir
echo
echo "DONE. See results at end of file $FABELSTREAM_BUILD/results.txt"

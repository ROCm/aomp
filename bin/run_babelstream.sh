#!/bin/bash

# run_babelstream.sh - runs babelstream in the $BABELSTREAM_BUILD dir.
# User can set RUN_OPTIONS to control which variants to build and run
# The variants are:
#    omp-default
#    omp-fast
#    hip
#    hip-um    - requires XNACK and sets HSA_XNACK=1
#    omp-cpu
#    stdpar    - requires XNACK and sets HSA_XNACK=1
#    stdind    - requires XNACK and sets HSA_XNACK=1
#    omp-mi300 - requires XNACK and sets HSA_XNACK=1
#    omp-usm   - requires XNACK and sets HSA_XNACK=1
#
# This script allows overrides to the following environment variables:
#   BABELSTREAM_BUILD   - Build directory with results /tmp/$USER/babelstream
#   BABELSTREAM_REPO    - Directory for the Babelstream repo
#   BABLESTREAM_REPEATS - The number of repititions, default is 10
#   AOMP                - Compiler install directory, can set to /opt/rocm/lib/llvm
#   AOMPHIP             - HIP Compiler install directory, defaults to $AOMP
#   RUN_OPTIONS         - which variants to build and execute
#                         default is "omp-default omp-fast hip stdpar"
#   ENABLE_USM=1        - run variants "omp-default omp-fast omp-mi300 omp-usm hip hip-um"
#
# --- Start standard header to set AOMP environment variables ----
realpath=`realpath $0`
thisdir=`dirname $realpath`
export AOMP_USE_CCACHE=0

export ROCR_VISIBLE_DEVICES=0

. $thisdir/aomp_common_vars
# --- end standard header ----

if [ "$1" != "nocopy" ] ; then
patchrepo $AOMP_REPOS_TEST/$AOMP_BABELSTREAM_REPO_NAME
fi

# Set defaults for environment variables
AOMP=${AOMP:-/usr/lib/aomp}
AOMPHIP=${AOMPHIP:-$AOMP}

# note: clone_test.sh puts babelstream in $AOMP_REPOS_TEST/babelstream
BABELSTREAM_REPO=${BABELSTREAM_REPO:-$AOMP_REPOS_TEST/babelstream}
BABELSTREAM_BUILD=${BABELSTREAM_BUILD:-/tmp/$USER/babelstream}
BABELSTREAM_REPEATS=${BABELSTREAM_REPEATS:-10}
BABELSTREAM_ARRAY_NUM_ELEMS=${BABELSTREAM_ARRAY_NUM_ELEMS:-''}

if [ ! -z ${BABELSTREAM_ARRAY_NUM_ELEMS} ]; then
  BABELSTREAM_ARRAY_SIZE_COMPUTED=$(( $BABELSTREAM_ARRAY_NUM_ELEMS * 1024))
  BABELSTREAM_ARRAY_SIZE="-s ${BABELSTREAM_ARRAY_SIZE_COMPUTED}"
fi

# Clean validation files
rm -f "$BABELSTREAM_REPO"/passing-tests.txt "$BABELSTREAM_REPO"/failing-tests.txt "$BABELSTREAM_REPO"/make-fail.txt

# Use function to set and test AOMP_GPU
setaompgpu
# skip hip if nvidida
if [ "${AOMP_GPU:0:3}" == "sm_" ] ; then
   TRIPLE="nvptx64-nvidia-cuda"
   RUN_OPTIONS=${RUN_OPTIONS:-"omp-default omp-fast"}
else
   TRIPLE="amdgcn-amd-amdhsa"
   if [ "$ENABLE_USM" == 1 ]; then
     RUN_OPTIONS=${RUN_OPTIONS:-"omp-default omp-fast omp-mi300 omp-usm hip hip-um"}
   else
     RUN_OPTIONS=${RUN_OPTIONS:-"omp-default omp-fast hip stdpar"}
   fi
fi

omp_target_flags="-O3 -fopenmp -fopenmp-targets=$TRIPLE -Xopenmp-target=$TRIPLE -march=$AOMP_GPU -DOMP -DOMP_TARGET_GPU -Dsimd="
#  So this script runs with old comilers, we only use -fopenmp-target-fast
#  for LLVM 16 or higher
LLVM_VERSION=`$AOMP/bin/clang++ --version | grep version | cut -d" " -f 3 | cut -d"." -f1`
if [ "$LLVM_VERSION" == "version" ]  ; then
  # If 3rd  arg is version, must be aomp or rocm compiler, so get version from 4th field.
  LLVM_VERSION=`$AOMP/bin/clang++ --version | grep version | cut -d" " -f 4 | cut -d"." -f1`
  special_aso_flags="-fopenmp-gpu-threads-per-team=1024 -fopenmp-target-fast"
  special_mi300_flags="-fopenmp-target-fast -fopenmp-target-fast-reduction"
else
  # temp hack to detect trunk (not vendor compiler) (found version in 3 args)"
  special_aso_flags="-fopenmp-assume-no-thread-state -fopenmp-assume-no-nested-parallelism -fopenmp-cuda-mode -Rpass=openmp-opt -Rpass-missed=openmp-opt -Rpass-analysis=openmp-opt"
fi
echo "LLVM VERSION IS $LLVM_VERSION"
_SILENT=""
# for old versions of gpurun without -s flag, silence gpurun with GPURUN_VERBOSE
export GPURUN_VERBOSE=0
if [[ $LLVM_VERSION -ge 18 ]] ; then
  omp_fast_flags="$special_aso_flags $omp_target_flags"
  omp_mi300_flags="$special_mi300_flags $omp_target_flags"
  _SILENT="-s"
else
 if [[ $LLVM_VERSION -ge 16 ]] ; then
  omp_fast_flags="$special_aso_flags $omp_target_flags"
  _SILENT="-s"
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
 omp_mi300_flags="$omp_fast_flags"
fi

GPURUN_BINDIR=${GPURUN_BINDIR:-$AOMP/bin}
if [ ! -f "$GPURUN_BINDIR/gpurun" ] || [ ! -f "$GPURUN_BINDIR/rocminfo" ] ; then
    # When using trunk, try to find gpurun and rocminfo in ROCm
    _SILENT=""
    GPURUN_BINDIR=/opt/rocm/llvm/bin
    export ROCMINFO_BINARY=/opt/rocm/bin/rocminfo
else
    export ROCMINFO_BINARY=$GPURUN_BINDIR/rocminfo
fi

omp_cpu_flags="-O3 -fopenmp -DOMP"
hip_flags="-O3 -Wno-unused-result -DHIP"
stdpar_flags="-O3 -I. --offload-arch=$AOMP_GPU -DNDEBUG -O3 --hipstdpar --hipstdpar-path=/opt/rocm/include/thrust/system/hip --hipstdpar-thrust-path=/opt/rocm/include --hipstdpar-prim-path=/opt/rocm/include -std=c++17 "

omp_src="main.cpp OMPStream.cpp"
hip_src="main.cpp HIPStream.cpp"
stdpar_src="main.cpp STDDataStream.cpp"
stdind_src="main.cpp STDIndicesStream.cpp"

gccver=`gcc --version | grep gcc | cut -d")" -f2 | cut -d"." -f1`
std="-std=c++20"

if [ ! -d $BABELSTREAM_REPO ]; then
  echo "ERROR: BabelStream not found in $BABELSTREAM_REPO"
  echo "       Consider running these commands:"
  echo
  echo "mkdir -p $AOMP_REPOS_TEST"
  echo "cd $AOMP_REPOS_TEST"
  echo "git clone https://github.com/UoB-HPC/babelstream"
  echo
  exit 1
fi
curdir=$PWD
mkdir -p $BABELSTREAM_BUILD
echo cd $BABELSTREAM_BUILD
cd $BABELSTREAM_BUILD
if [ "$1" != "nocopy" ] ; then
   cp $BABELSTREAM_REPO/src/main.cpp .
   cp $BABELSTREAM_REPO/src/Stream.h .
   cp $BABELSTREAM_REPO/src/omp/OMPStream.cpp .
   cp $BABELSTREAM_REPO/src/omp/OMPStream.h .
   cp $BABELSTREAM_REPO/src/hip/HIPStream.cpp .
   cp $BABELSTREAM_REPO/src/hip/HIPStream.h .
   cp $BABELSTREAM_REPO/src/std-data/STDDataStream.cpp .
   cp $BABELSTREAM_REPO/src/std-data/STDDataStream.h .
   cp $BABELSTREAM_REPO/src/std-indices/STDIndicesStream.cpp .
   cp $BABELSTREAM_REPO/src/std-indices/STDIndicesStream.h .
   cp $BABELSTREAM_REPO/src/dpl_shim.h .
else
   # for nocopy option, ensure temp sources exist (possibly edited),
   # AND save a copies of unpatched upstream code in temp dir for easy compare.
   [ ! -f OMPStream.cpp ] && echo missing $BABELSTREAM_BUILD/OMPStream.cpp && exit 1
   cp $BABELSTREAM_REPO/src/omp/OMPStream.cpp OMPStream.cpp.orig
   [ ! -f HIPStream.cpp ] && echo missing $BABELSTREAM_BUILD/HIPStream.cpp && exit 1
   cp $BABELSTREAM_REPO/src/hip/HIPStream.cpp HIPStream.cpp.orig
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
    compile_cmd="$AOMP/bin/clang++ $std $omp_target_flags $omp_src -o $EXEC"
  elif [ "$option" == "omp-fast" ]; then
    compile_cmd="$AOMP/bin/clang++ $std $omp_fast_flags   $omp_src -o $EXEC"
  elif [ "$option" == "omp-mi300" ]; then
    compile_cmd="$AOMP/bin/clang++ $std $omp_mi300_flags   $omp_src -o $EXEC"
  elif [ "$option" == "omp-cpu" ]; then
    compile_cmd="$AOMP/bin/clang++ $std $omp_cpu_flags    $omp_src -o $EXEC"
  elif [ "$option" == "omp-usm" ]; then
    compile_cmd="$AOMP/bin/clang++ -DOMP_USM $std $omp_fast_flags   $omp_src -o $EXEC"
  elif [ "$option" == "hip" ] || [ "$option" == "hip-um" ] || [ "$option" == "stdpar" ] || [ "$option" == "stdind" ] ; then
    if [ ! -f $AOMPHIP/bin/hipcc ] ; then 
      AOMPHIP="$AOMPHIP/.."
      if [ ! -f $AOMPHIP/bin/hipcc ] ; then
        # if $AOMP is trunk, try to use hipcc from rocm
        if [ -f /opt/rocm/bin/hipcc ] ; then
          AOMPHIP=/opt/rocm
        else
          echo "ERROR: $AOMPHIP/bin/hipcc not found"
          exit 1
        fi
      fi
    fi
    if [ "$option" == "stdpar" ]; then
      compile_cmd="$AOMPHIP/bin/hipcc -DSTD_DATA $stdpar_flags $stdpar_src -o $EXEC"
    elif [ "$option" == "stdind" ]; then
      compile_cmd="$AOMPHIP/bin/hipcc -DSTD_INDICES $stdpar_flags $stdind_src -o $EXEC"
    else
      compile_cmd="$AOMPHIP/bin/hipcc $std $hip_flags $hip_src -o $EXEC"
    fi
    HIP_PATH=$AOMPHIP
  else
    echo "ERROR: Option not recognized: $option."
    exit 1
  fi
  echo $compile_cmd | tee -a results.txt
  $compile_cmd
  if [ $? -ne 0 ]; then
    compile_error=1
    echo "babelstream" >> "$BABELSTREAM_REPO"/make-fail.txt
    break
  else
    set -o pipefail
    if [ -f $GPURUN_BINDIR/gpurun ] ; then
      if [ "$option" == "omp-usm" ]; then
         echo HSA_XNACK=1 $GPURUN_BINDIR/gpurun $_SILENT ./$EXEC -n $BABELSTREAM_REPEATS ${BABELSTREAM_ARRAY_SIZE} | tee -a results.txt
         HSA_XNACK=1 $GPURUN_BINDIR/gpurun $_SILENT ./$EXEC -n $BABELSTREAM_REPEATS ${BABELSTREAM_ARRAY_SIZE} 2>&1 | tee -a results.txt
      elif [ "$option" == "hip-um" ]; then
         echo HSA_XNACK=1 HIP_UM=1 $GPURUN_BINDIR/gpurun $_SILENT ./$EXEC -n $BABELSTREAM_REPEATS ${BABELSTREAM_ARRAY_SIZE} | tee -a results.txt
         HSA_XNACK=1 HIP_UM=1 $GPURUN_BINDIR/gpurun $_SILENT ./$EXEC -n $BABELSTREAM_REPEATS ${BABELSTREAM_ARRAY_SIZE} 2>&1 | tee -a results.txt
      elif [ "$option" == "stdpar" ]; then
         echo HSA_XNACK=1 $GPURUN_BINDIR/gpurun $_SILENT ./$EXEC -n $BABELSTREAM_REPEATS ${BABELSTREAM_ARRAY_SIZE} | tee -a results.txt
         HSA_XNACK=1 $GPURUN_BINDIR/gpurun $_SILENT ./$EXEC -n $BABELSTREAM_REPEATS ${BABELSTREAM_ARRAY_SIZE} | tee -a results.txt
      elif [ "$option" == "stdind" ]; then
         echo HSA_XNACK=1 $GPURUN_BINDIR/gpurun $_SILENT ./$EXEC -n $BABELSTREAM_REPEATS ${BABELSTREAM_ARRAY_SIZE} | tee -a results.txt
         HSA_XNACK=1 $GPURUN_BINDIR/gpurun $_SILENT ./$EXEC -n $BABELSTREAM_REPEATS ${BABELSTREAM_ARRAY_SIZE} | tee -a results.txt
      elif [ "$option" == "omp-fast" ]; then
         echo echo OMPX_FORCE_SYNC_REGIONS=1 \
                $GPURUN_BINDIR/gpurun $_SILENT ./$EXEC -n $BABELSTREAM_REPEATS ${BABELSTREAM_ARRAY_SIZE} 2>&1 | tee -a results.txt
         OMPX_FORCE_SYNC_REGIONS=1 \
                $GPURUN_BINDIR/gpurun $_SILENT ./$EXEC -n $BABELSTREAM_REPEATS ${BABELSTREAM_ARRAY_SIZE} 2>&1 | tee -a results.txt
      else
         echo $GPURUN_BINDIR/gpurun $_SILENT ./$EXEC -n $BABELSTREAM_REPEATS ${BABELSTREAM_ARRAY_SIZE} | tee -a results.txt
         $GPURUN_BINDIR/gpurun $_SILENT ./$EXEC -n $BABELSTREAM_REPEATS ${BABELSTREAM_ARRAY_SIZE} 2>&1 | tee -a results.txt
      fi
      if [ $? -ne 0 ]; then
        runtime_error=1
      fi
    else
      echo ./$EXEC -n $BABELSTREAM_REPEATS ${BABELSTREAM_ARRAY_SIZE} | tee -a results.txt
      ./$EXEC -n $BABELSTREAM_REPEATS ${BABELSTREAM_ARRAY_SIZE} 2>&1 | tee -a results.txt
      if [ $? -ne 0 ]; then
        runtime_error=1
      fi
    fi
  fi
done

# Check for errors
if [ $compile_error -ne 0 ]; then
  echo "babelstream" >> "$BABELSTREAM_REPO"/make-fail.txt
elif [ $runtime_error -ne 0 ]; then
  echo "babelstream" >> "$BABELSTREAM_REPO"/failing-tests.txt
else
  echo "babelstream" >> "$BABELSTREAM_REPO"/passing-tests.txt
fi

cd $curdir
echo
echo "DONE. See results at end of file $BABELSTREAM_BUILD/results.txt"
if [ "$1" != "nocopy" ] ; then
removepatch $AOMP_REPOS_TEST/$AOMP_BABELSTREAM_REPO_NAME
fi

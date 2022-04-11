#!/bin/bash

# run_og11_babelstream.sh - compile with OG11 g++  and run babelstream

OG11DIR=${OG11DIR:-$HOME/git/og11/install}
OG11GXX=${OG11GXX:-$OG11DIR/bin/g++}
ROCMDIR=${ROCMDIR:-/opt/rocm}
GPURUN=$HOME/rocm/aomp/bin/gpurun
DO_CPU_RUNS=0
DO_OVERRIDES=0

AOMP_REPOS=${AOMP_REPOS:-$HOME/git/aomp15.0}
AOMP_REPOS_TEST=${AOMP_REPOS_TEST:-$HOME/git/aomp-test}
BABELSTREAM_REPO=${BABELSTREAM_REPO:-$AOMP_REPOS_TEST/babelstream}
BABELSTREAM_PATCH=${BABELSTREAM_PATCH:-$AOMP_REPOS/aomp/og11/og11_babelstream.patch}
BABELSTREAM_BUILD=${BABELSTREAM_BUILD:-/tmp/$USER/babelstream}
OFFLOAD_ARCH_BIN=${OFFLOAD_ARCH_BIN:-$ROCMDIR/llvm/bin/offload-arch}

if [ ! -f $OFFLOAD_ARCH_BIN ] ; then 
  echo "ERROR: ROCM binary missing: $OFFLOAD_ARCH_BIN" 
  exit 1
fi
_offload_arch=`$OFFLOAD_ARCH_BIN`

if [ ! -d $BABELSTREAM_REPO ]; then
  echo "ERROR: BabelStream not found in $BABELSTREAM_REPO"
  echo "       Consider running these commands:"
  echo
  echo "cd"
  echo "git clone https://github.com/UoB-HPC/babelstream"
  echo
  exit 1
fi
if [ ! -f $BABELSTREAM_PATCH ]; then
  echo "ERROR: BabelStream patch not found in $BABELSTREAM_PATCH"
  exit 1
fi
curdir=$PWD
mkdir -p $BABELSTREAM_BUILD
echo cd $BABELSTREAM_BUILD
cd $BABELSTREAM_BUILD
if [ "$1" != "nopatch" ] ; then 
   cp $BABELSTREAM_REPO/src/main.cpp .
   cp $BABELSTREAM_REPO/src/Stream.h .
   cp $BABELSTREAM_REPO/src/omp/OMPStream.cpp .
   cp $BABELSTREAM_REPO/src/omp/OMPStream.cpp OMPStream.orig.cpp
   cp $BABELSTREAM_REPO/src/omp/OMPStream.h .
   echo "Applying patch: $BABELSTREAM_PATCH"
   patch < $BABELSTREAM_PATCH
fi
omp_src="main.cpp OMPStream.cpp"
omp_src_orig="main.cpp OMPStream.orig.cpp"
thisdate=`date`
echo | tee a results.txt
echo "=========> RUNDATE:  $thisdate" | tee -a results.txt
echo "=========> GPU:      $_offload_arch" || tee -a results.txt
echo | tee -a results.txt
echo "=========> NAME:     1 og11 OFFLOAD DEFAULTS" | tee -a results.txt
echo "=========> COMPILER: $OG11GXX" | tee -a results.txt
og11_flags="-O3 -fopenmp -foffload=-march=$_offload_arch -D_OG11_DEFAULTS -DOMP -DOMP_TARGET_GPU"
   export LD_LIBRARY_PATH=$OG11DIR/lib64:$ROCMDIR/hsa/lib
   export OMP_TARGET_OFFLOAD=MANDATORY
   unset GCN_DEBUG
   #export GCN_DEBUG=1
   EXEC=omp-stream-og11
   rm -f $EXEC
   if [ -f $GPURUN ] ; then 
      echo $GPURUN $OG11GXX $og11_flags $omp_src -o $EXEC | tee -a results.txt
      $GPURUN $OG11GXX $og11_flags $omp_src -o $EXEC
   else
      echo $OG11GXX $og11_flags $omp_src -o $EXEC | tee -a results.txt
      $OG11GXX $og11_flags $omp_src -o $EXEC
   fi
   if [ $? -ne 1 ]; then
     ./$EXEC 2>&1 | tee -a results.txt
   fi
   unset OMP_TARGET_OFFLOAD

echo | tee -a results.txt
echo "=========> NAME:     2. clang no simd" | tee -a results.txt
_clangcc="$ROCMDIR/llvm/bin/clang++"
_clang_omp_flags="-std=c++11 -O3 -fopenmp -fopenmp-targets=amdgcn-amd-amdhsa -Xopenmp-target=amdgcn-amd-amdhsa -march=$_offload_arch -DOMP -DOMP_TARGET_GPU -D_DEFAULTS_NOSIMD"
echo "=========> COMPILER: $_clangcc" | tee -a results.txt
   export OMP_TARGET_OFFLOAD=MANDATORY
   EXEC=omp-stream-clang
   rm -f $EXEC
   echo $_clangcc $_clang_omp_flags $omp_src -o $EXEC | tee -a results.txt
   $_clangcc $_clang_omp_flags $omp_src -o $EXEC
   if [ $? -ne 1 ]; then
     ./$EXEC 2>&1 | tee -a results.txt
   fi

if [ $DO_OVERRIDES  == 1 ] ; then 
   echo | tee -a results.txt
   echo "=========> NAME:     3 og11 OVERRIDES" | tee -a results.txt
   echo "=========> COMPILER: $OG11GXX" | tee -a results.txt
   og11_flags="-O3 -fopenmp -foffload=-march=$_offload_arch -D_OG11_OVERRIDE -DNUM_THREADS=1024 -DNUM_TEAMS=240 -DOMP -DOMP_TARGET_GPU"
   export LD_LIBRARY_PATH=$OG11DIR/lib64:$ROCMDIR/hsa/lib
   export OMP_TARGET_OFFLOAD=MANDATORY
   unset GCN_DEBUG
   #export GCN_DEBUG=1
   EXEC=omp-stream-og11-overrides
   rm -f $EXEC
   echo $OG11GXX $og11_flags $omp_src -o $EXEC | tee -a results.txt
   $OG11GXX $og11_flags $omp_src -o $EXEC
   if [ $? -ne 1 ]; then
     ./$EXEC 2>&1 | tee -a results.txt
   fi
   unset OMP_TARGET_OFFLOAD
fi

if [ $DO_CPU_RUNS == 1 ] ; then 
   echo | tee -a results.txt
   echo "=========> NAME:     4. og11 OPENMP CPU" | tee -a results.txt
   echo "=========> COMPILER: $OG11GXX" | tee -a results.txt
   omp_flags_cpu="-O3 -fopenmp -DOMP"
   unset OMP_TARGET_OFFLOAD
   export LD_LIBRARY_PATH=$OG11DIR/lib64:$ROCMDIR/hsa/lib
   EXEC=omp-stream-og11-cpu
   rm -f $EXEC
   echo $OG11GXX $omp_flags_cpu $omp_src -o $EXEC
   $OG11GXX $omp_flags_cpu $omp_src -o $EXEC
   if [ $? -ne 1 ]; then
     ./$EXEC 2>&1 | tee -a results.txt
   fi

   echo | tee -a results.txt
   echo "=========> NAME:     5. gcc OPENMP CPU" | tee -a results.txt
   GXX_VERSION=`g++ --version | grep -m1 "g++" | awk '{print $4}'`
   echo "=========> COMPILER: g++ $GXX_VERSION" | tee -a results.txt
   omp_flags_cpu="-O3 -fopenmp -DOMP"
   unset OMP_TARGET_OFFLOAD
   unset LD_LIBRARY_PATH
   EXEC=omp-stream-gxx-cpu
   rm -f $EXEC
   echo g++ $omp_flags_cpu $omp_src_orig -o $EXEC | tee -a results.txt
   g++ $omp_flags_cpu $omp_src_orig -o $EXEC
   if [ $? -ne 1 ]; then
     ./$EXEC 2>&1 | tee -a results.txt
   fi
fi

echo
echo "DONE. See results at end of file $BABELSTREAM_BUILD/results.txt"
cd $curdir

#!/bin/bash
# Copyright © Advanced Micro Devices, Inc., or its affiliates.
#
#  run_theRockCI.sh
#    runs smoke, smoke-fort, smoke-limbo, smoke-fort-libmo,
#    babelstream fortran-babelstream accel2023 hpc2021 openmpapps
#    override with SUITE_LIST
#  Please check with Ron or Ethan for script modifications.

SUITE_LIST=${SUITE_LIST:-"smoke-limbo smoke-fort-limbo smoke smoke0firt nekbone babelstream fortran-babelstream accel2023 hpc2021 openmpapps"}

export PATH=$PATH:/opt/rocm/bin
echo "PATH=" $PATH
set +x
which lspci
which rocm-smi
which rocminfo
which make

export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/opt/rocm/llvm/lib
echo "LD_LIBRARY_PATH="$LD_LIBRARY_PATH
set -x
pip install --no-warn-script-location filecheck
export PATH=$PATH:/home/$USER/.local/bin
which filecheck

export INST=/tmp/npsdbInst$$/openmpi-5-npsdb
export MPI=$INST
echo rocmMPI=$MPI

RUN_SPEC=1
WLOC=https://compute-artifactory.amd.com/artifactory/rocm-generic-local/compiler-infra
wget --timeout 15 --tries=3  $WLOC/Accel23-scripts.tar
if [ "$?" -ne 0 ]; then
  echo "SPECScripts not accessible " $?
  RUN_SPEC=0
else
  echo "SPECscripts are available"
fi
 
./rocm_quick_check.sh
export ROCR_VISIBLE_DEVICES=0
export AOMP_USE_CCACHE=0

echo $SUITE_LIST

TLOG=/tmp/log$$
echo "================"  >$TLOG

# Use bogus path to avoid using target.lst, a user-defined target list
# used by rocm_agent_enumerator.
export ROCM_TARGET_LST=/opt/nowhere

#ulimit -t 1000

realpath=`realpath $0`
scriptdir=`dirname $realpath`
parentdir=`eval "cd $scriptdir;pwd;cd - > /dev/null"`
aompdir="$(dirname "$parentdir")"
summary=`pwd`"/summary.txt"
scriptfails=0

EPSDB=1 ./clone_test.sh  
AOMP_TEST_DIR=${AOMP_TEST_DIR:-"$HOME/git/aomp-test"}
echo AOMP before : $AOMP
if [ ! -e $AOMP/bin ]; then
  echo $AOMP does not point to valid location, unsetting
  unset AOMP
fi
# Set AOMP to point to rocm symlink or newest version.
if [ -e /opt/rocm/lib/llvm/bin ]; then
  AOMP=${AOMP:-"/opt/rocm/lib/llvm"}
  ROCMINF=/opt/rocm
  ROCMDIR=/opt/rocm/lib
  echo setting 1 $AOMP
elif [ -e /opt/rocm/llvm/bin ]; then
  AOMP=${AOMP:-"/opt/rocm/llvm"}
  ROCMINF=/opt/rocm
  ROCMDIR=/opt/rocm
  echo setting 2 $AOMP
else
echo "error 1"
exit
fi
export AOMP
echo "AOMP = $AOMP"

if [ ! -f "$AOMP/bin/gpurun" ]; then
  echo "Error: Could not find gpurun"
  exit 1
# rm -f "$HOME/openmp-utils/bin/gpurun"
# if ! wget -P "$HOME/openmp-utils/bin" https://compute-artifactory.amd.com/artifactory/rocm-generic-local/compiler-infra/gpurun ; then
#   echo "Error: Could not download gpurun"
#   exit 1
# fi
# chmod 755 "$HOME/openmp-utils/bin/gpurun"
# export GPURUN_BINDIR="$HOME/openmp-utils/bin"
# export PATH=$PATH:$GPURUN_BINDIR
fi

clangversion=`$AOMP/bin/clang --version`
aomp=0
if [[ "$clangversion" =~ "AOMP_STANDALONE" ]]; then
  aomp=1
fi

# Make sure clang is present.
$AOMP/bin/clang --version
if [ $? -ne 0 ]; then
  echo "Error: Clang not found at "$AOMP"/bin/clang."
  exit 1
fi

$AOMP/bin/flang --version
if [ $? -ne 0 ]; then
  echo "Error: flang not found at "$AOMP"/bin/flang."
  exit 1
fi

# Use bogus path to avoid using target.lst, a user-defined target list
# used by rocm_agent_enumerator.
export ROCM_TARGET_LST=/opt/nowhere

echo "AMDGPU devices:"
$ROCMINF/bin/rocm_agent_enumerator

# Set AOMP_GPU.
# Regex skips first result 'gfx000' and selects second id.
if [ "$AOMP_GPU" == "" ]; then
  AOMP_GPU=$($ROCMINF/bin/rocm_agent_enumerator | grep -m 1 -E gfx[^0]{1}.{2})
fi

# mygpu will eventually relocate to /opt/rocm/bin, support both cases for now.
if [ "$AOMP_GPU" != "" ]; then
  echo "AOMP_GPU set with rocm_agent_enumerator."
else
  echo "AOMP_GPU is empty, use mygpu."
  if [ -a $AOMP/bin/mygpu ]; then
    AOMP_GPU=$($AOMP/bin/mygpu)
  else
    AOMP_GPU=$($AOMP/../bin/mygpu)
  fi
fi
if [ "$AOMP_GPU" == "" ]; then
  echo "Error: AOMP_GPU was not able to be set with RAE or mygpu."
  exit 1
fi
echo AOMP_GPU=$AOMP_GPU
export AOMP_GPU

# Run quick sanity test
echo
echo "check-xnack test"
cd "$aompdir"/test/smoke-limbo/check-xnack
make clean > /dev/null
VERBOSE=1 make
./check-xnack
HSA_XNACK=1 OMPX_APU_MAPS=1 ./check-xnack
echo
echo "Helloworld sanity test:"
cd "$aompdir"/test/smoke/helloworld
make clean > /dev/null
OMP_TARGET_OFFLOAD=MANDATORY VERBOSE=1 make run > hello.log 2>&1
sed -n -e '/ld.lld/,$p' hello.log
echo
echo "Checking plugin"
LIBOMPTARGET_DEBUG=1 OMP_TARGET_OFFLOAD=MANDATORY make run 2>&1 | grep "libomptarget.rtl.amdgpu"
echo

function smoke-fort(){
  echo "%================ smoke-fort"
  cd "$aompdir"/test/smoke-fort
  ./check_smoke_fort.sh
  if [ "$?" -eq "0" ]; then
     echo "Passed smoke-fort">> $TLOG
  else 
     echo "FAILED smoke-fort">> $TLOG
     scriptfails=1  
  fi
}

function smoke(){
  echo "%================ smoke"
  cd "$aompdir"/test/smoke
  ./check_smoke.sh
  if [ "$?" -eq "0" ]; then
     echo "Passed smoke" >> $TLOG
  else 
     echo "FAILED smoke" >> $TLOG
     scriptfails=1  
  fi
}

function smoke-fort-limbo(){
  echo "%================ smoke-fort-limbo"
  cd "$aompdir"/test/smoke-fort-limbo
  ./check_smoke_fort_limbo.sh
  if [ "$?" -eq "0" ]; then
     echo "Passed smoke-fort_limbo" >> $TLOG
  else 
     echo "FAILED smoke-fort_limbo" >> $TLOG
     scriptfails=1  
  fi
}

function smoke-limbo(){
  echo "%================ smoke-limbo"
  cd "$aompdir"/test/smoke-limbo
  ./check_smoke_limbo.sh
  if [ "$?" -eq "0" ]; then
     echo "Passed smoke-limbo" >> $TLOG
  else 
     echo "FAILED smoke-limbo" >> $TLOG
     scriptfails=1  
  fi
}

function openmpapps(){
  echo "%================ openmpapps"
  # -----Run Openmpapps-----
  cd "$AOMP_TEST_DIR"/openmpapps
  echo rockMPI=$MPI
  ./check_openmpapps.sh
  if [ "$?" -eq "0" ]; then
     echo "Passed openmpapps" >> $TLOG
  else 
     echo "FAILED openmpapps" >> $TLOG
     scriptfails=1  
  fi
}

function nekbone(){
  echo "%================ nekbone"
  # -----Run Nekbone-----
  cd "$aompdir"/bin
  ( VERBOSE=0 ./run_nekbone.sh ) 
  if [ "$?" -eq "0" ]; then
     echo "Passed Nekbone" >> $TLOG
  else 
     echo "FAILED Nekbone" >> $TLOG
     scriptfails=1  
  fi
}

function babelstream(){
  echo "%================ baelstream"
  export AOMPHIP=$ROCMDIR
  cd "$aompdir"/bin
  if [ $aomp -eq 0 ]; then
    export ROCMINFO_BINARY=$ROCMINF/bin/rocminfo
  fi
  export RUN_OPTIONS="omp-default omp-fast"
  ./run_babelstream.sh
  if [ "$?" -eq "0" ]; then
     echo "Passed Babelstream" >> $TLOG
  else 
     echo "FAILED Babelstream" >> $TLOG
     scriptfails=1  
  fi
}

function fortran-babelstream(){
  echo "%================ fortran-babelstream"
  export AOMPHIP=$ROCMDIR
  cd "$aompdir"/bin
  if [ $aomp -eq 0 ]; then
    export ROCMINFO_BINARY=$ROCMINF/bin/rocminfo
  fi
  ./run_fBabel.sh
  if [ "$?" -eq "0" ]; then
     echo "Passed fortran-babelstream" >> $TLOG
  else 
     echo "FAILED fortran-babelstream" >> $TLOG
     scriptfails=1  
  fi
}


function accel2023(){
echo "%================ accel2023"
  if [ "$RUN_SPEC" -eq 0 ]; then
    echo "Skipping accel2023, runners external to AMD"
    exit 0
  fi
  cd "$aompdir"/bin
  export GPURUN_BYPASS=1
  ./run_accel2023.sh -clean
  cd $AOMP_TEST_DIR/accel2023-2.0.18
  grep ratio= result/*.log  | tail -12
  nsucc=$(grep ratio= result/*.log  | grep Succ | wc -l)
  if [ $nsucc -eq 12 ]; then
    echo "Passed accel2023 $nsucc passes"  >> $TLOG
  else
    echo "FAILED accel2023 $nsucc passes"  >> $TLOG
     scriptfails=1  
  fi
}

function hpc2021(){
echo "%================ hpc2021"
  if [ "$RUN_SPEC" -eq 0 ]; then
    echo "Skipping hpc2021, runners external to AMD"
    exit 0
  fi
  cd "$aompdir"/bin
  unset ROCR_VISIBLE_DEVICES
  export GPURUN_BYPASS=1
  echo rockMPI=$MPI
  ./run_hpc2021.sh -clean
  cd $AOMP_TEST_DIR/hpc2021-1.1.9
  grep ratio= result/*.log  | tail -9
  nsucc=$(grep ratio= result/*.log  | grep Succ | wc -l)
  if [ $nsucc -eq 9 ]; then
    echo "Passed hpc2021 $nsucc passes"  >> $TLOG
  else
    echo "FAILED hpc2021 $nsucc passes"  >> $TLOG
     scriptfails=1  
  fi
}

echo Running List: $SUITE_LIST

declare -A warnings
warningcount=0
for suite in $SUITE_LIST; do
  $suite
done

echo "************************************" > $summary
if [ "$scriptfails" != 0 ]; then
  echo FAIL >> $summary
  echo "EPSDB Status:  red" >> $summary
else
  echo PASS >> $summary
  echo "EPSDB Status:  green" >> $summary
fi
cat $TLOG
echo ""
echo >> $summary
cat $summary
exit $((scriptfails))

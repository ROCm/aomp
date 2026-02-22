#!/bin/bash
# Copyright © Advanced Micro Devices, Inc., or its affiliates.
#
#  run_theRockCI.sh
#    runs smoke, smoke-fort, smoke-limbo, smoke-fort-libmo,
#    babelstream fortran-babelstream accel2023 hpc2021 openmpapps
#    override with SUITE_LIST
#  Please check with Ron or Ethan for script modifications.
date
SUITE_LIST=${SUITE_LIST:-"smoke-limbo smoke-fort-limbo smoke smoke-fort nekbone babelstream fortran-babelstream accel2023 bldopenmpi hpc2021 openmpapps"}
declare -A assocSuite=(
["smoke-limbo"]=" 5 minutes"
["smoke-fort-limbo"]=" 2 minutes"
["smoke"]=" 14 minutes"
["smoke-fort"]=" 5 minutes"
["nekbone"]=" 1 minute"
["babelstream"]=" 1 minute"
["fortran-babelstream"]=" 1 minute"
["accel2023"]=" 3 minutes"
["bldopenmpi"]=" 6 minutes"
["hpc2021"]=" 4 minutes"
["openmpapps"]=" 2 minutes"
)

function hide_unrunnable_files() {
pushd $aompdir/test/smoke/
mkdir .savem
mv clang-host-targ clang-host-targ2 flags host_targ .savem/
popd
pushd $aompdir/test/smoke-fort
mkdir .savem
mv flang-gpu-abort .savem/
popd
}


ulimit -t 400
ulimit -s unlimited

tmpfile=/tmp/smoke-$$
export PATH=$PATH:/opt/rocm/bin
echo "PATH=" $PATH
which lspci
which rocm-smi
which rocminfo
which make

export GPURUN_BYPASS=1
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/opt/rocm/llvm/lib
echo "LD_LIBRARY_PATH="$LD_LIBRARY_PATH
pip install --no-warn-script-location filecheck
export FILECHECK=`which filecheck`
echo filecheck: $FILECHECK

RUN_SPEC=1
WLOC=https://compute-artifactory.amd.com/artifactory/rocm-generic-local/compiler-infra
wget --timeout 5 --tries=2  $WLOC/Accel23-scripts.tar
if [ "$?" -ne 0 ]; then
  echo "SPECScripts not accessible " $?
  RUN_SPEC=0
else
  echo "SPECscripts are available"
fi

if [ "$SKIP_QUICK" == "" ]; then
./rocm_quick_check.sh
fi
export ROCR_VISIBLE_DEVICES=0
export AOMP_USE_CCACHE=0

echo $SUITE_LIST

TLOG=/tmp/log$$
echo "================"  >$TLOG

# Use bogus path to avoid using target.lst, a user-defined target list
# used by rocm_agent_enumerator.
export ROCM_TARGET_LST=/opt/nowhere


realpath=`realpath $0`
scriptdir=`dirname $realpath`
parentdir=`eval "cd $scriptdir;pwd;cd - > /dev/null"`
aompdir="$(dirname "$parentdir")"
summary=`pwd`"/summary.txt"
scriptfails=0

if [ "$SKIP_CLONE" == "" ]; then
  EPSDB=1 ./clone_test.sh
fi
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
  pushd $AOMP
  cd `realpath .`
  cd ../..
  ROCMINF=`pwd`/
  ROCMDIR=`pwd`
  popd
fi
export AOMP
echo "AOMP = $AOMP"

if [ ! -f "$AOMP/bin/gpurun" ]; then
  echo "Error: Could not find gpurun"
  exit 1
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

function checkRes() {
  tail -100 $1
  actual=`grep "Passing tests: " $1 | awk -F'[ /\033]' '{print $4}'`
  expect=`grep "Passing tests: " $1 | awk -F'[ /\033]' '{print $5}'`
  if [ "$actual" == "$expect" ]; then
    return 0;
  else
    return 1;
  fi
}

function smoke-fort(){
  echo "%================ smoke-fort"
  cd "$aompdir"/test/smoke-fort
  ./check_smoke_fort.sh > $tmpfile 2>&1
  checkRes $tmpfile
  if [ "$?" == 0 ]; then
     echo "Passed smoke-fort">> $TLOG
  else
     echo "FAILED smoke-fort">> $TLOG
  fi
}

function smoke(){
  echo "%================ smoke"
  cd "$aompdir"/test/smoke
  ./check_smoke.sh > $tmpfile 2>&1
  checkRes $tmpfile
  if [ "$?" == 0 ]; then
     echo "Passed smoke" >> $TLOG
  else
     echo "FAILED smoke" >> $TLOG
  fi
}

function smoke-fort-limbo(){
  echo "%================ smoke-fort-limbo"
  cd "$aompdir"/test/smoke-fort-limbo
  ./check_smoke_fort_limbo.sh > $tmpfile 2>&1
  checkRes $tmpfile
  if [ "$?" == 0 ]; then
     echo "Passed smoke-fort-limbo" >> $TLOG
  else
     echo "FAILED smoke-fort-limbo" >> $TLOG
  fi
}

function smoke-limbo(){
  echo "%================ smoke-limbo"
  cd "$aompdir"/test/smoke-limbo
  export SKIP_OMPT=1
  ./check_smoke_limbo.sh > $tmpfile 2>&1
  unset SKIP_OMPT
  checkRes $tmpfile
  if [ "$?" == 0 ]; then
     echo "Passed smoke-limbo" >> $TLOG
  else
     echo "FAILED smoke-limbo" >> $TLOG
  fi
}

function openmpapps(){
  echo "%================ openmpapps"
  # -----Run Openmpapps-----
  cd "$AOMP_TEST_DIR"/openmpapps
  rm -rf matmul_usm hpgmg-mp4
  echo rockMPI=$MPI
  ./check_openmpapps.sh
  git checkout .
  if [ "$?" == 0 ]; then
     echo "Passed openmpapps" >> $TLOG
  else
     echo "FAILED openmpapps" >> $TLOG
  fi
}

function nekbone(){
  echo "%================ nekbone"
  # -----Run Nekbone-----
  cd "$aompdir"/bin
  ( VERBOSE=0 ./run_nekbone.sh )
  if [ "$?" == 0 ]; then
     echo "Passed Nekbone" >> $TLOG
  else
     echo "FAILED Nekbone" >> $TLOG
  fi
}

function babelstream(){
  echo "%================ babelestream"
  export AOMPHIP=$ROCMDIR
  cd "$aompdir"/bin
  if [ $aomp -eq 0 ]; then
    export ROCMINFO_BINARY=$ROCMINF/bin/rocminfo
  fi
  export RUN_OPTIONS="omp-default omp-fast"
  ./run_babelstream.sh
  if [ "$?" == 0 ]; then
     echo "Passed Babelstream" >> $TLOG
  else
     echo "FAILED Babelstream" >> $TLOG
  fi
}

function fortran-babelstream(){
  echo "%================ fortran-babelstream"
  export AOMPHIP=$ROCMDIR
  cd "$aompdir"/bin
  if [ "$?" != 0 ]; then
    export ROCMINFO_BINARY=$ROCMINF/bin/rocminfo
  fi
  ./run_fBabel.sh
  if [ "$?" == 0 ]; then
     echo "Passed fortran-babelstream" >> $TLOG
  else
     echo "FAILED fortran-babelstream" >> $TLOG
  fi
}


function accel2023(){
  echo "%================ accel2023"
  if [ "$RUN_SPEC" -eq 0 ]; then
    echo "Skipping accel2023, runners external to AMD"
    return 0
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
  fi
}

function hpc2021(){
  echo "%================ hpc2021"
  if [ "$RUN_SPEC" -eq 0 ]; then
    echo "Skipping hpc2021, runners external to AMD"
    return 0
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
  fi
}

function bldopenmpi(){
  echo "%================ OpenMPI"
  export NO_HPC2021_MPI_BLD=1
  export INST=${INST:-/tmp/npsdbInst$$/openmpi-5-flang}
  export MPI=$INST
  echo rocmMPI=$MPI
  pushd $aompdir/bin
  ./npsdb_bld_ompi.sh
  popd
}

echo Running List: $SUITE_LIST

hide_unrunnable_files

declare -A warnings
warningcount=0
for suite in $SUITE_LIST; do
  echo "=== Running $suite `date` ==="
  echo "--- expected time: ${assocSuite[$suite]}"
  if [[ "$suite" =~ "smoke" ]]; then
    $suite  2>&1 |tail -200
  else
    $suite
  fi
done

echo "************************************" > $summary
grep -i -q fail $TLOG
scriptfails=$?
if [ "$scriptfails" == 0 ]; then
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
date
exit $((scriptfails))

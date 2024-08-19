#!/bin/bash
#--------------------------------------
# WARNING: Intended for developers to use with AOMP standalone Git Hub releases.
#
# run_test_suite.sh
#  Available Options:
#  --Groups--
#  ./run_test_suite.sh                   -run all tests (default): $SUITE_LIST
#  ./run_test_suite.sh epsdb             -run epsdb focused subset of tests: $EPSDB_LIST
#
#  --Individual Suites--"
#  ./run_test_suite.sh smoke             -run single suite (smoke) only
#  ./run_test_suite.sh smoke examples    -run user specifed multiple suites

# Disabled Suites
DISABLED_LIST="raja omptests"

# Available Suites - Qmcpack will timeout at 20 minutes
SUITE_LIST=${SUITE_LIST:-"ovo examples smoke hipopenmp omp5 openmpapps nekbone openmp_vv llnl openlibm qmcpack"}

#Groups
GROUP_LIST="epsdb"
EPSDB_LIST=${EPSDB_LIST:-"examples smoke hipopenmp omp5 openmpapps nekbone openmp_vv"}

# Set up variables
AOMP_REPOS=${AOMP_REPOS:-"$HOME/git/aomp19.0"}
AOMP_SRC=${AOMP_SRC:-$AOMP_REPOS/aomp}
AOMP_BIN=${AOMP_BIN:-$AOMP_SRC/bin}
AOMP_REPOS_TEST=${AOMP_REPOS_TEST:-"$HOME/git/aomp-test"}
AOMP=${AOMP:-"$HOME/rocm/aomp"}

echo "AOMP_REPOS: $AOMP_REPOS"
echo "AOMP_SRC:   $AOMP_SRC"
echo "AOMP_REPOS_TEST:  $AOMP_REPOS_TEST"
echo "AOMP set:   $AOMP"
echo ""

# Source common vars, set AOMP_GPU
cd $AOMP_BIN
. aomp_common_vars
export NUM_THREADS
export AOMP
setaompgpu

# Log directories/files
function create_logs(){
  log_dir=$AOMP_SRC/test/test-suite-results/$(date '+%b-%d-%Y')/$(date '+%H-%M-%S')
  mkdir -p $log_dir
  results_file=results-report.log
  echo Final Log: $log_dir/$results_file
}

function header(){
  echo Running $1...
  {
    echo "------------------------------------------------------------"
    echo "                           $1                               "
    echo "------------------------------------------------------------"
  } >> $log_dir/$results_file
}

function update_logs(){
  # $1 - bin/cmd/log
  # i.e.
  #  bin nekbone
  #  cmd make test
  #  log results.txt
  #  log resuilts.txt 30

  if [ "$1" == "bin" ]; then
    shift
    ./$@ >> $log_dir/$results_file 2>&1
  elif [ "$1" == "cmd" ]; then
    shift
    echo $@
    $@ >> $log_dir/$results_file 2>&1
  elif [ "$1" == "log" ]; then
    shift
    if [ -f $1 ]; then
      if [ ! -z $2 ] && [ $2 -gt 0 ]; then
        tail -$2 $1 >> $log_dir/$results_file
      else
        cat $1 >> $log_dir/$results_file
      fi
    else
      echo Error: Specified file does not exist: $1
    fi
  else
    echo Error: Command not recognized: $@
  fi
}

function cmake_warning(){
  echo "-----------------------------------------------------------------------------------------------------"
  echo "Warning!! It is recommended to build Raja with a cmake version between 3.9 and and 3.16.8 (inclusive)."
  echo "Raja may fail to build otherwise."
  echo "HIT ENTER TO CONTINUE or CTRL-C TO CANCEL"
  echo "-----------------------------------------------------------------------------------------------------"
  read
}

function check_cmake(){
# Check cmake version
if [ "$CHECK_CMAKE" != 0 ]; then
  cmake_regex="(([0-9])+\.([0-9]+)\.[0-9]+)"
  cmake_ver_str=$($AOMP_CMAKE --version)
  if [[ "$cmake_ver_str" =~ $cmake_regex ]]; then
    cmake_ver=${BASH_REMATCH[1]}
    cmake_major_ver=${BASH_REMATCH[2]}
    cmake_minor_ver=${BASH_REMATCH[3]}
    echo "Cmake found: version $cmake_ver"
    if [ "$cmake_major_ver" != "3" ]; then
      cmake_warning
    elif (( $(echo "$cmake_minor_ver > 16"| bc -l) ||  $(echo "$cmake_minor_ver < 9" | bc -l) )); then
      cmake_warning
    fi
  else
    echo "ERROR: No cmake found, exiting..."
    exit 1
  fi
fi
}

# --------Begin Suites--------
function examples(){
  # -----Run Examples-----
  header EXAMPLES
  cd $AOMP_SRC/examples
  echo "Log file at: $log_dir/examples.log"
  ./check_examples.sh > $log_dir/examples.log 2>&1
  for directory in ./*/; do
    # Raja/Kokkos not executed here
    if [[ $directory =~ "raja" ]] || [[ $directory =~ "kokkos" ]]; then
      continue
    fi
    pushd $directory > /dev/null
    path=$(pwd) && base=$(basename $path)
    if [ -f check-$base.txt ]; then
      update_logs log check-$base.txt
      echo "" >> $log_dir/$results_file
    fi
    popd > /dev/null
  done
}

function smoke(){
  # -----Run Smoke-----
  header SMOKE
  cd $AOMP_SRC/test/smoke > /dev/null
  echo "Log file at: $log_dir/smoke.log"
  AOMP_PARALLEL_SMOKE=1 CLEANUP=0 ./check_smoke.sh > $log_dir/smoke.log 2>&1
  update_logs bin check_smoke.sh gatherdata
}

function smokefails(){
  # -----Run Smoke-----
  header SMOKEFAILS
  cd $AOMP_SRC/test/smoke-fails > /dev/null
  echo "Log file at: $log_dir/smoke-fails.log"
  ./check_smoke_fails.sh > $log_dir/smoke-fails.log 2>&1
  update_logs bin check_smoke_fails.sh gatherdata
}

function hipopenmp(){
  # -----Run Hip-openmp-----
  header HIP-OPENMP
  cd $AOMP_SRC/test/hip-openmp > /dev/null
  echo "Log file at: $log_dir/hip-openmp.log"
  ./check_hip-openmp.sh > $log_dir/hip-openmp.log 2>&1
  update_logs log hip-openmp-check.txt
}

function usm(){
 # -----Run usm-----
  header usm
  cd $AOMP_SRC/test/usm > /dev/null
  echo "Log file at: $log_dir/usm.log"
  ./check_usm.sh > $log_dir/usm.log 2>&1
  update_logs bin check_usm.sh
}

function omp5(){
 # -----Run Omp5-----
  header OMP5
  cd $AOMP_SRC/test/omp5 > /dev/null
  echo "Log file at: $log_dir/omp5.log"
  ./check_omp5.sh > $log_dir/omp5.log 2>&1
  update_logs bin check_omp5.sh gatherdata
}

function openmpapps(){
  # -----Run Openmpapps-----
  header OPENMPAPPS
  cd $AOMP_REPOS_TEST/openmpapps
  git pull
  echo "Log file at: $log_dir/openmpapps.log"
  ./check_openmpapps.sh > $log_dir/openmpapps.log 2>&1
  update_logs log times-openmpapps.txt
  update_logs log check-openmpapps.txt
}

function nekbone(){
  # -----Run Nekbone-----
  header NEKBONE
  cd $AOMP_SRC/bin
  echo "Log file at: $log_dir/nekbone.log"
  ./run_nekbone.sh > $log_dir/nekbone.log 2>&1

  cd $AOMP_REPOS_TEST/Nekbone/test/nek_gpu1
  ulimit -s unlimited
  update_logs bin nekbone
}

function openmp_vv(){
  # -----Run OpenMP_VV (formerly SOLLVE) -----
  header OPENMP_VV
  cd $AOMP_REPOS_TEST/openmp_vv
  git pull

  cd $AOMP_SRC/bin
  echo "Log file at: $log_dir/openmp_vv.log"
  ./run_openmp_vv.sh > $log_dir/openmp_vv.log 2>&1

  cd $AOMP_REPOS_TEST/openmp_vv
  update_logs log combined-results.txt
}

# Skipped for now, RAJA is causing GPU issues and requires
# a system restart to reset the GPU.
function raja(){
  # -----Run RAJA-----
  header RAJA
  export AOMP_CMAKE=${AOMP_CMAKE:-cmake}
  echo "Log file at: $log_dir/raja.log"
  cd $AOMP_SRC/examples/raja/vecadd
  make clean > $log_dir/raja.log 2>&1
  make clean_raja >> $log_dir/raja.log 2>&1
  make run >> $log_dir/raja.log 2>&1
  if [ $? -eq 0 ]; then
    raja_build_dir=$(ls $HOME | grep raja_build_omp)
    if [ $? -eq 0 ]; then
      cd $HOME/$raja_build_dir
      ARGS="--timeout 30" update_logs cmd make test
    else
      echo Error: Raja build directory not found. >> $log_dir/raja.log 2>&1
    fi
  else
    echo Error: Raja make was not successful. >> $log_dir/raja.log 2>&1
  fi
}

function kokkos(){
  # -----Run KOKKOS-----
  header KOKKOS
  export AOMP_CMAKE=${AOMP_CMAKE:-cmake}
  echo "Log file at: $log_dir/kokkos.log"
  cd $AOMP_SRC/examples/kokkos/helloworld
  make clean > $log_dir/kokkos.log 2>&1
  make clean_kokkos >> $log_dir/kokkos.log 2>&1
  make run >> $log_dir/kokkos.log 2>&1
  if [ $? -eq 0 ]; then
    kokkos_build_dir=$(ls $HOME | grep kokkos_build_omp.$AOMP_GPU$)
    if [ $? -eq 0 ]; then
      cd $HOME/$kokkos_build_dir
      ARGS="--timeout 120" update_logs cmd make test
    else
      echo "Error: Kokkos build directory not found." >> $log_dir/kokkos.log 2>&1
    fi
  else
    echo "Error: Kokkos make was not successful." >> $log_dir/kokkos.log 2>&1
  fi
}

function llnl(){
  # -----Run LLNL-----
  header LLNL
  echo "Log file at: $log_dir/LLNL.log"
  cd $AOMP_SRC/test/LLNL/openmp5.0-tests
  ./check_LLNL.sh >> $log_dir/LLNL.log 2>&1
  update_logs log $log_dir/LLNL.log 3
}

function ovo(){
  # -----Run OvO-----
  header OvO
  cd $AOMP_SRC/bin
  echo "Log file at: $log_dir/ovo.log"
  ./run_ovo.sh >> $log_dir/ovo.log 2>&1
  cd $AOMP_REPOS_TEST/OvO
  update_logs bin ovo.sh report --summary
  # Report returns 1 due to test failures
  return 0
}

function openlibm(){
  # -----Run openlibm-----
  header OPENLIBM
  cd $AOMP_SRC/bin
  echo "Log file at: $log_dir/openlibm.log"
  ./run_openlibm_test.sh >> $log_dir/openlibm.log 2>&1
  cat $log_dir/openlibm.log | grep -E "(Openlibm test)|(Total tests)|(Errors)|(Passing tests)|(Pass rate)|(Logfile)" >> $log_dir/$results_file
}

function qmcpack(){
  # -----Run QMCPACK-----
  set -e
  header QMCPACK
  cd $AOMP_SRC/bin
  echo "Log file at: $log_dir/qmcpack.log"
  timeout --foreground -k 35m 35m ./build_qmcpack.sh  >> $log_dir/qmcpack.log 2>&1
  build_dir=$AOMP_REPOS_TEST/qmcpack/build_AOMP_offload_real_MP_$AOMP_GPU
  set +e
  if [ $? -eq 0 ]; then
    if [ -d $build_dir ];then
      echo "cd $build_dir"
      cd $build_dir
      update_logs cmd ctest -R deterministic
    else
      echo "build_dir: $build_dir not found!" >> $log_dir/$results_file
      echo "build_dir: $build_dir not found!"
    fi
  fi
}

function omptests(){
  #-----Run IBM omptests-----
  header OMPTESTS
  cd $AOMP_SRC/bin
  echo "Log file at: $log_dir/omptests.log"
  ./run_omptests.sh >> $log_dir/omptests.log 2>&1
  update_logs log $log_dir/omptests.log 8
}
# --------End Suites--------

function print_help(){
  echo
  echo "--------Run Test Suite Help-------"
  echo
  echo "WARNING: Intended for developers to use with AOMP standalone Git Hub releases."
  echo "NOTICE:  RAJA and IBM omptests disabled for now as they were locking up the GPU."
  echo
  echo "All Test Suites:   $SUITE_LIST"
  echo "EPSDB Test Suites: $EPSDB_LIST"
  echo
  echo "User can modify SUITE_LIST env variable to choose which suites to run:"
  echo "SUITE_LIST=\"smoke smoke-fails\" ./run_test_suite.sh"
  echo
  echo "Available Options:"
  echo "  --Groups--"
  echo "  ./run_test_suite.sh                   -run all tests (default): $SUITE_LIST"
  echo "  ./run_test_suite.sh epsdb             -run epsdb focused subset of tests: $EPSDB_LIST"
  echo
  echo "  --Individual Suites--"
  echo "  ./run_test_suite.sh smoke             -run single suite (smoke) only"
  echo "  ./run_test_suite.sh smoke examples    -run user specifed multiple suites"
  echo
  exit
}


# Subsets
function epsdb(){
  for suite in $EPSDB_LIST; do
    $suite
  done
}

function check_args(){
  for arg in $@; do
    if [ "$arg" == "raja" ]; then
      check_cmake
    fi
    arg_regex="(^$arg| $arg$|$arg | $arg )"
    #Parse individual suites and groups
    if [[ ! "$SUITE_LIST" =~ $arg_regex ]] && [[ ! "$GROUP_LIST" =~ $arg_regex ]]; then
      echo "Invalid argument: \"$arg\", printing help..."
      print_help
      exit 1
    fi
  done
}

# Main run function
function run_tests(){
  # Custom test list
  if [ $# -ne 0 ]; then
    echo Running Custom List: $@
    for arg in $@; do
      $arg
      if [ $? -ne 0 ]; then
        if [ -f "$log_dir/$log_file"$arg".log" ]; then
          echo Inspect $log_dir/$log_file"$arg".log for build output. >> $log_dir/$results_file
        fi
      fi
      echo "" >> $log_dir/$results_file
    done
  # Default SUITE_LIST
  else
    echo Running List: $SUITE_LIST
    for suite in $SUITE_LIST; do
      $suite
      echo "" >> $log_dir/$results_file
    done
  fi
}

# Execute tests
  if [ "$1" == "-h" ] || [[ "$1" =~ "help" ]]; then
    print_help
  else
    check_args $@
    create_logs
    run_tests $@
  fi

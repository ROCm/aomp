#!/bin/bash
#
# Checks all tests in smoke-fails directory using make check. 
# Programs return 0 for success or a number > 0 for failure.
# Tests that need to be visually inspected: devices, pfspecify, pfspecify_str, stream
#
#

#Text Colors
RED="\033[0;31m"
GRN="\033[0;32m"
BLU="\033[0;34m"
ORG="\033[0;33m"
BLK="\033[0m"

function gatherdata(){
  #Gather Test Data
  passing_tests=0
  if [ -e passing-tests.txt ]; then
    ((passing_tests=$(wc -l <  passing-tests.txt)))
    total_tests=$passing_tests
  fi
  if [ -e make-fail.txt ]; then
    ((total_tests+=$(wc -l <  make-fail.txt)))
  fi
  if [ -e failing-tests.txt ]; then
    ((total_tests+=$(wc -l <  failing-tests.txt)))
  fi

  #Print Results
  echo -e "$BLU"-------------------- Results --------------------"$BLK"
  echo -e "$BLU"Number of tests: $total_tests"$BLK"
  echo ""
  echo -e "$GRN"Passing tests: $passing_tests/$total_tests""

  #Print passing tests, if any
  if [ -e passing-tests.txt ]; then
    echo "--------------------"
    cat passing-tests.txt
  fi

  #Print failed tests
  echo -e "$RED"
  echo ""
  if [ -e failing-tests.txt ]; then
    echo "Runtime Fails"
    echo "--------------------"
    cat failing-tests.txt
    echo ""
  fi

  if [ -e make-fail.txt ]; then
    echo "Compile Fails"
    echo "--------------------"
    cat make-fail.txt
  fi
  echo -e "$BLK"
}

if [ "$1" == "gatherdata" ]; then
  gatherdata
  exit 0
fi

cleanup(){
  rm -f passing-tests.txt
  rm -f failing-tests.txt
  rm -f check-smoke.txt
  rm -f make-fail.txt
}

script_dir=$(dirname "$0")
pushd $script_dir
path=$(pwd)

#Clean all testing directories
cleanup

if [ "$1" == "-clean" ]; then
  for directory in ./*/; do
    pushd $directory > /dev/null
    make clean
    popd > /dev/null
  done
  exit 0
fi

export OMP_TARGET_OFFLOAD=${OMP_TARGET_OFFLOAD:-MANDATORY}
echo OMP_TARGET_OFFLOAD=$OMP_TARGET_OFFLOAD

echo ""
echo -e "$ORG"RUNNING ALL TESTS IN: $path"$BLK"
echo ""

echo "************************************************************************************" > check-smoke.txt
echo "                   A non-zero exit code means a failure occured." >> check-smoke.txt
echo "***********************************************************************************" >> check-smoke.txt

skip_tests=""
if [ "$SKIP_FORTRAN" == 1 ] ; then
  skip_tests+="`find .  -iname '*.f9[50]' | sed s^./^^ | awk -F/ '{print $1}'` "
  echo $skip_tests
fi

# ---------- Begin parallel logic ----------
if [ "$AOMP_PARALLEL_SMOKE" == 1 ]; then
  sem --help > /dev/null
  if [ $? -eq 0 ]; then
    COMP_THREADS=1
    MAX_THREADS=16
    if [ ! -z `which "getconf"` ]; then
       COMP_THREADS=$(`which "getconf"` _NPROCESSORS_ONLN)
       if [ "$AOMP_PROC" == "ppc64le" ] ; then
          COMP_THREADS=$(( COMP_THREADS / 6))
       fi
       if [ "$AOMP_PROC" == "aarch64" ] ; then
          COMP_THREADS=$(( COMP_THREADS / 4))
       fi
    fi
    AOMP_JOB_THREADS=${AOMP_JOB_THREADS:-$COMP_THREADS}
    if [ $AOMP_JOB_THREADS -gt 16 ]; then
      AOMP_JOB_THREADS=16
      echo "Limiting job threads to $AOMP_JOB_THREADS."
    fi
    echo THREADS: $AOMP_JOB_THREADS
    # Parallel Make
    for directory in ./*/; do
      pushd $directory > /dev/null
      base=$(basename `pwd`)
      echo Make: $base
      if [ $base == "gpus" ]; then # Compile and link only test
        make clean > /dev/null
        make &> make-log.txt
        if [ $? -ne 0 ]; then
          flock -e lockfile -c "echo $base: Make Failed >> ../make-fail.txt"
        else
          flock -e lockfile -c "echo $base  >> ../passing-tests.txt"
        fi
      else
        sem --jobs $AOMP_JOB_THREADS  --id def_sem -u 'base=$(basename $(pwd)); make clean > /dev/null; make &> make-log.txt; if [ $? -ne 0 ]; then flock -e lockfile -c "echo $base: Make Failed >> ../make-fail.txt"; fi;'
      fi
      popd > /dev/null
    done

    # Wait for jobs to finish before execution
    sem --wait --id def_sem

    # Parallel execution, currently limited to 4 jobs
    for directory in ./*/; do
      pushd $directory > /dev/null
      base=$(basename `pwd`)
      echo RUN $base
      if [ $base == 'hip_rocblas' ] ; then
        ls $AOMPROCM/rocblas > /dev/null 2>&1
        if [ $? -ne 0 ]; then
          echo -e "$RED"$base - needs rocblas installed at $AOMPROCM/rocblas:"$BLK"
          echo -e "$RED"$base - ROCBLAS NOT FOUND!!! SKIPPING TEST!"$BLK"
          popd > /dev/null
          continue
        fi
      elif [ $base == 'devices' ] || [ $base == 'stream' ] ; then
        sem --jobs 4 --id def_sem -u 'make run > /dev/null 2>&1'
        sem --jobs 4 --id def_sem -u 'make check > /dev/null 2>&1'
      elif [ $base == 'printf_parallel_for_target' ] || [ $base == 'omp_places' ] || [ $base == 'pfspecifier' ] || [ $base == 'pfspecifier_str' ] ; then
        sem --jobs 4 --id def_sem -u 'make verify-log > /dev/null'
      elif [ $base == 'flags' ] ; then
        make run
      elif [ $base == 'liba_bundled' ] || [ $base == 'liba_bundled_cmdline' ]; then
        sem --jobs 4 --id def_sem -u 'base=$(basename $(pwd)); make check > /dev/null; if [ $? -ne 0 ]; then flock -e lockfile -c "echo $base: Make Failed >> ../make-fail.txt"; fi;'
      elif [ $base == "gpus" ]; then # Compile and link only test
        echo gpus is compile only!
      else
        sem --jobs 4 --id def_sem -u 'make check > /dev/null 2>&1'
      fi
      popd > /dev/null
    done

    # Wait for jobs to finish executing
    sem --wait --id def_sem
    gatherdata
    exit
  else
    echo
    echo "Warning: Parallel smoke requested, but the parallel package needed is not installed. Continuing with sequential version..."
    echo
  fi
fi
# ---------- End parallel logic ----------


#Loop over all directories and make run / make check depending on directory name
for directory in ./*/; do
  (cd "$directory" && path=$(pwd) && base=$(basename $path)
  if [ "$NO_CLEAN" != 1 ] ; then	
    make clean
  fi
  #Skip tests that are known failures
  skip=0
  for test in $skip_tests ; do
    if [ $test == $base ] ; then
      skip=1
      echo "$test $base"
      break
    fi
  done
  if [ $skip -ne 0 ] ; then
    echo "Skipping $base"
    echo ""
  else  
    AOMPROCM=${AOMPROCM:-/opt/rocm}
    if [ $base == 'hip_rocblas' ] ; then
      ls $AOMPROCM/rocblas > /dev/null 2>&1
      if [ $? -ne 0 ]; then
        echo -e "$RED"$base - needs rocblas installed at $AOMPROCM/rocblas:"$BLK"
        echo -e "$RED"$base - ROCBLAS NOT FOUND!!! SKIPPING TEST!"$BLK"
        continue
      fi
    fi
    if [ $base == 'devices' ] || [ $base == 'pfspecifier' ] || [ $base == 'pfspecifier_str' ] || [ $base == 'stream' ] ; then
      make
      if [ $? -ne 0 ]; then
        echo "$base: Make Failed" >> ../make-fail.txt
      fi
      make run > /dev/null 2>&1
      make check > /dev/null 2>&1

    #flags has multiple runs
    elif [ $base == 'flags' ] ; then
      make
      make run > /dev/null 2>&1
    elif [ $base == 'printf_parallel_for_target' ] ; then
      make verify-log
    else
      make
      if [ $? -ne 0 ]; then
        echo "$base: Make Failed" >> ../make-fail.txt
      fi
      make check > /dev/null 2>&1
      #liba_bundled has an additional Makefile, that may fail on the make check
      if [ $? -ne 0 ] && ( [ $base == 'liba_bundled' ] || [ $base == 'liba_bundled_cmdline' ] ) ; then
        echo "$base: Make Failed" >> ../make-fail.txt
      fi
    fi
  fi
  echo ""
 )
	
done

#Print run.log for all tests that need visual inspection
for directory in ./*/; do
  (cd "$directory" && path=$(pwd) && base=$(basename $path)
    if [ $base == 'devices' ] || [ $base == 'pfspecifier' ] || [ $base == 'pfspecifier_str' ] || [ $base == 'stream' ] ; then
      echo ""
      echo -e "$ORG"$base - Run Log:"$BLK"
      echo "--------------------------"
      if [ -e run.log ]; then
        cat run.log
      fi
      echo ""
      echo ""
    fi
  )
done

#Replace false positive return codes with 'Check the run.log' so that user knows to visually inspect those.
sed -i '/pfspecifier/ {s/0/Check the run.log above/}; /devices/ {s/0/Check the run.log above/}; /stream/ {s/0/Check the run.log above/}' check-smoke.txt
echo ""
if [ -e check-smoke.txt ]; then
  cat check-smoke.txt
fi
if [ -e make-fail.txt ]; then
  cat make-fail.txt
fi
echo ""

gatherdata

# Print run logs for runtime fails, EPSDB only
# Disable printing of runtime fails for smoke-fails
if [ "$EPSDB" == 2 ] ; then
  file='failing-tests.txt'
  flags_test_done=0
  if [ -e $file ]; then
    echo ----------Printing Runtime Fail Logs---------
    while read -r line; do
      # The flags test has multiple numbered runs. We cannot pushd flags 1 because only the flags dir exists.
      # We must re-run the entire flags test to get run.log. If more than one flags subtest fails only run once.
      if [[ "$line" =~ "flags" ]]; then
        if [[ "$flags_test_done" == 0 ]]; then
          echo
          pushd flags > /dev/null
          echo Test: flags run log:
          echo The flags test must run all iterations if one subtest fails.
          make run
          cat run.log
          flags_test_done=1
          popd > /dev/null
        fi
      else
        echo
        pushd $line > /dev/null
        echo
        make run > /dev/null
        echo Test: $line run log:
        cat run.log
        popd > /dev/null
      fi
    done < "$file"
    echo
  fi
fi

popd

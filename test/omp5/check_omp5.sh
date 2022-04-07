#!/bin/bash
#
# Checks all tests in omp5 directory using make check. Programs return 0 for success or a number > 0 for failure.
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
  echo -e "$GRN"Passing tests: $passing_tests/$total_tests"$BLK"
  echo ""

  #Print failed tests
  echo -e "$RED"
  if [ "$SKIP_FAILS" != 1 ] && [ "$known_fails" != "" ] ; then
    echo "Known failures: $known_fails"
  fi
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
  rm -f check-omp5.txt
  rm -f make-fail.txt
}

path=$(pwd)

#Clean all testing directories
make clean
cleanup

echo ""
echo -e "$ORG"RUNNING ALL TESTS IN: $path"$BLK"
echo ""

echo "************************************************************************************" > check-omp5.txt
echo "                   A non-zero exit code means a failure occured." >> check-omp5.txt
echo "***********************************************************************************" >> check-omp5.txt

known_fails="red_bug_51 shape_noncontig concur_update mapper_prob loop prag_error"

if [ "$SKIP_FAILURES" == 1 ] ; then
  skip_tests=$known_fails
else
  skip_tests=""
fi

#Loop over all directories and make run / make check depending on directory name
for directory in ./*/; do
    (cd "$directory" && path=$(pwd) && base=$(basename $path)
    #Skip tests that are known failures
    skip=0
    for test in $skip_tests ; do
      if [ $test == $base ] ; then
        skip=1
        break
      fi
    done
    if [ $skip -ne 0 ] ; then
      echo "Skip $base!"
      echo ""
      continue
    fi
    if [ $base == 'hip_rocblas' ] ; then
      ls /opt/rocm/rocblas > /dev/null 2>&1
      if [ $? -ne 0 ]; then
        echo -e "$RED"$base - needs rocblas installed at /opt/rocm/rocblas:"$BLK"
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
sed -i '/pfspecifier/ {s/0/Check the run.log above/}; /devices/ {s/0/Check the run.log above/}; /stream/ {s/0/Check the run.log above/}' check-omp5.txt
echo ""
if [ -e check-omp5.txt ]; then
  cat check-omp5.txt
fi
if [ -e make-fail.txt ]; then
  cat make-fail.txt
fi
echo ""

gatherdata

#Tests that need visual inspection
#echo ""
#echo -e "$ORG"
#echo "---------- Please inspect the output above to verify the following tests ----------"
#echo "stream"
#echo -e "$BLK"

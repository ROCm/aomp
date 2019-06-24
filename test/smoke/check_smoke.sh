#!/bin/bash
#
# Checks all tests in smoke directory using make check. Programs return 0 for success or a number > 0 for failure.
# Tests that need to be visually inspected: devices, pfspecify, pfspecify_str, stream
#
#

#Text Colors
RED="\033[0;31m"
GRN="\033[0;32m"
BLU="\033[0;34m"
ORG="\033[0;33m"
BLK="\033[0m"

path=$(pwd)

#Clean all testing directories
make clean
rm passing-tests.txt
rm failing-tests.txt
rm check-smoke.txt
rm make-fail.txt

echo ""
echo -e "$ORG"RUNNING ALL TESTS IN: $path"$BLK"
echo ""

echo "************************************************************************************" > check-smoke.txt
echo "                   A non-zero exit code means a failure occured." >> check-smoke.txt
echo "Tests that need to be visually inspected: devices, pfspecify, pfspecify_str, stream" >> check-smoke.txt
echo "***********************************************************************************" >> check-smoke.txt

#Loop over all directories and make run / make check depending on directory name
for directory in ./*/; do
  (cd "$directory" && path=$(pwd) && base=$(basename $path)
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
      if [ $? -ne 0 ] && [ $base == 'liba_bundled' ] ; then
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
      cat run.log
      echo ""
      echo ""
    fi
  )
done

#Replace false positive return codes with 'Check the run.log' so that user knows to visually inspect those.
sed -i '/pfspecifier/ {s/0/Check the run.log above/}; /devices/ {s/0/Check the run.log above/}; /stream/ {s/0/Check the run.log above/}' check-smoke.txt
echo ""
cat check-smoke.txt
cat make-fail.txt
echo ""

#Gather Test Data
((total_tests=$(wc -l <  passing-tests.txt)))
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
echo -e "$GRN"Passing tests: `wc -l <  passing-tests.txt`/$total_tests"$BLK"
echo ""

#Print failed tests
echo -e "$RED"
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

#Tests that need visual inspection
echo ""
echo -e "$ORG"
echo "---------- Please inspect the output above to verify the following tests ----------"
echo "devices"
echo "pfspecifier"
echo "pfspecifier_str"
echo "stream"
echo -e "$BLK"

#Clean up, hide output
rm check-smoke.txt
rm passing-tests.txt
if [ -e failing-tests.txt ]; then
  rm failing-tests.txt
fi
if [ -e make-fail.txt ]; then
  rm make-fail.txt
fi

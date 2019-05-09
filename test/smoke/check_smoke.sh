#!/bin/bash
#
# Checks all tests in smoke directory using make check. Programs return 0 for success or a number > 0 for failure.
# Tests that need to be visually inspected: devices, pfspecify, pfspecify_str, stream
#
#
path=$(pwd)
count=0
allTests=""
failedTests=""
tempDir=""

#Clean all testing directories
make clean
rm passing-tests.txt
rm check-smoke.txt
rm make-fail.txt

echo ""
echo "RUNNING ALL TESTS IN: $path "
echo ""

echo "************************************************************************************" > check-smoke.txt
echo "                   A non-zero exit code means a failure occured." >> check-smoke.txt
echo "Tests that need to be visually inspected: devices, pfspecify, pfspecify_str, stream" >> check-smoke.txt
echo "***********************************************************************************" >> check-smoke.txt

#Loop over all directories and make run / make check depending on directory name
for directory in ./*/; do
        shopt -s lastpipe
        echo -n "$directory " | tr -d ./ | read tempDir
        tempDir+=" "
        allTests+="$tempDir"
	let count++
	(cd "$directory" && path=$(pwd) && base=$(basename $path) 
		if [ $base == 'devices' ] || [ $base == 'pfspecifier' ] || [ $base == 'pfspecifier_str' ] || [ $base == 'stream' ] ; then 
			make
                        if [ $? -ne 0 ]; then
                                echo "$base: Make Failed" >> ../make-fail.txt
                        fi
			make run > /dev/null 2>&1
		        make check > /dev/null 2>&1
		else
			make
                        if [ $? -ne 0 ]; then
                                echo "$base: Make Failed" >> ../make-fail.txt
                        fi
			make check > /dev/null 2>&1
		fi
	)
	
done

#Print run.log for all tests that need visual inspection
for directory in ./*/; do
	(cd "$directory" && path=$(pwd) && base=$(basename $path)
		if [ $base == 'devices' ] || [ $base == 'pfspecifier' ] || [ $base == 'pfspecifier_str' ] || [ $base == 'stream' ] ; then 
			echo ""
			echo "$base - Run Log:"
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

#Clean all testing directories, hide output
make clean > /dev/null 2>&1

echo "-------------------- Results --------------------"
echo "Number of tests: "$count
echo ""
echo "Number of passing tests: " `wc -l <  passing-tests.txt`"/"$count
echo ""

#If the test is not found in the passing-tests file, add test to failedTests
for test in $allTests
do
        if [ "$(grep -cx "$test" passing-tests.txt)" -eq 0 ];
        then
                failedTests+="$test "
        fi
done

#Print failed tests
echo "Failed Tests:"
for fail in $failedTests
do
        echo "$fail"
done

#Tests that need visual inspection
echo ""
echo "---------- Please inspect the output above to verify the following tests ----------"
echo "devices"
echo "pfspecifier"
echo "pfspecifier_str"
echo "stream"

#Clean up, hide output
make clean > /dev/null 2>&1
rm check-smoke.txt
rm passing-tests.txt
rm make-fail.txt

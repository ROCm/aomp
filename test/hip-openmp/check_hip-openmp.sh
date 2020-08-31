#!/bin/bash
#
# Checks all tests in smoke directory using make check. Programs return 0 for success or a number > 0 for failure.
# Tests for visual debugging: aomp_mappings, aomp_mappings_simple
# Tests that need to be visually inspected: devices, pfspecify, pfspecify_str, stream
#
#
echo ""
echo ""
path=$(pwd)
base=$(basename $path)
echo ""
echo "RUNNING ALL TESTS IN: $path "
echo ""

echo "" > hip-openmp-check.txt
echo "" >> hip-openmp-check.txt
echo "*******A non-zero exit code means a failure occured.*******" >> hip-openmp-check.txt
echo "*************************************************************" >> hip-openmp-check.txt

known_fails="aomp_hip_launch_test matmul_hip_omp_printf_fails matrixmul_omp_target"

#Loop over all directories and make run / make check depending on directory name
for directory in ./*/; do 
	(cd "$directory" && path=$(pwd) && base=$(basename $path) 
		echo "=== $base "
		make clean
		make	
		make run
		echo " Return Code for $base: $?" >> ../hip-openmp-check.txt
		make clean	
		
	)
	
done

#Print run.log for all tests that need visual inspection
for directory in ./*/; do
	(cd "$directory" && path=$(pwd) && base=$(basename $path)
	)
done

echo
echo "Known Fails: $known_fails"
cat hip-openmp-check.txt

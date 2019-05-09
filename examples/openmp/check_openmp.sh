#/bin/bash
#
# Checks all tests in examples/openmp directory using make check. Programs return 0 for success or a number > 0 for failure.
# Tests that need to be visually inspected: reduction
#
echo ""
echo ""
path=$(pwd)
base=$(basename $path)
echo ""
echo "RUNNING ALL TESTS IN: $path "
echo ""

echo "" > check-openmp.txt
echo "" >> check-openmp.txt
echo "*******A non-zero exit code means a failure occured.*******" >> check-openmp.txt
echo "*******Tests that need visual inspection: reduction********" >> check-openmp.txt
echo "***********************************************************" >> check-openmp.txt

#Loop over all directories and make run / make check depending on directory name
for directory in ./*/; do 
	(cd "$directory" && path=$(pwd) && base=$(basename $path) 
		make clean
		make
		make run
		echo " Return Code for $base: $?" >> ../check-openmp.txt
		make clean	
		
	)
	
done

#Print run.log for all tests that need visual inspection
for directory in ./*/; do
	(cd "$directory" && path=$(pwd) && base=$(basename $path)
	)
done
sed -i '/reduction/ {s/0/Check the output for success message./}' check-openmp.txt
cat check-openmp.txt

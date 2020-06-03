#/bin/bash
#
# Checks all tests in examples/openmp directory using make check. Programs return 0 for success or a number > 0 for failure.
# Tests that need to be visually inspected: reduction
#
echo ""
echo ""

script_dir=$(dirname "$0")
pushd $script_dir
path=$(pwd)

echo ""
echo "RUNNING ALL TESTS IN: $path "
echo ""

echo "" > check-openmp.txt
echo "" >> check-openmp.txt
echo "Openmp Results:" >> check-openmp.txt
echo "*******A non-zero exit code means a failure occured.*******" >> check-openmp.txt
echo "*******Tests that need visual inspection: reduction********" >> check-openmp.txt
echo "***********************************************************" >> check-openmp.txt

#Loop over all directories and make run / make check depending on directory name
for directory in ./*/; do 
	(cd "$directory" && path=$(pwd) && base=$(basename $path) 
		make clean
		make
		if [ $? -ne 0 ]; then
			echo "$base: Make Failed" >> ../check-openmp.txt
		else
			make run
			echo " Return Code for $base: $?" >> ../check-openmp.txt
		fi
		make clean	
		
	)
	
done

#Print run.log for all tests that need visual inspection
for directory in ./*/; do
	(cd "$directory" && path=$(pwd) && base=$(basename $path)
	)
done
cat check-openmp.txt
popd

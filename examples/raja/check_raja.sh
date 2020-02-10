#/bin/bash
#
# Checks all tests in examples/raja directory. Programs return 0 for success or a number > 0 for failure.
#
echo ""
echo ""
path=$(pwd)
base=$(basename $path)
echo ""
echo "RUNNING ALL TESTS IN: $path "
echo ""

echo "" > check-raja.txt
echo "" >> check-raja.txt
echo "Raja Results:" >> check-raja.txt
echo "*******A non-zero exit code means a failure occured.*******" >> check-raja.txt
echo "***********************************************************" >> check-raja.txt

#Loop over all directories and make run / make check depending on directory name
for directory in ./*/; do 
	(cd "$directory" && path=$(pwd) && base=$(basename $path) 
		make clean
    make clean_raja
		make
		make run
		echo " Return Code for $base: $?" >> ../check-raja.txt	
		make clean
    make clean_raja
	)
	
done

#Print run.log for all tests that need visual inspection
for directory in ./*/; do
	(cd "$directory" && path=$(pwd) && base=$(basename $path)
	)
done
cat check-raja.txt

#/bin/bash
#
# Checks all tests in hip directory using make make run. Programs return 0 for success or a number > 0 for failure.
#
echo ""
echo ""
path=$(pwd)
base=$(basename $path)
echo ""
echo "RUNNING ALL TESTS IN: $path "
echo ""

echo "" > check-hip.txt
echo "" >> check-hip.txt
echo "Hip Results:" >> check-hip.txt
echo "*******A non-zero exit code means a failure occured.*******" >> check-hip.txt
echo "***********************************************************" >> check-hip.txt

#Loop over all directories and make run / make check depending on directory name
for directory in ./*/; do 
	(cd "$directory" && path=$(pwd) && base=$(basename $path) 
		make clean
		make
		make run
		echo " Return Code for $base: $?" >> ../check-hip.txt
		make clean	
		
	)
	
done

#Print run.log for all tests that need visual inspection
for directory in ./*/; do
	(cd "$directory" && path=$(pwd) && base=$(basename $path)
	)
done
cat check-hip.txt

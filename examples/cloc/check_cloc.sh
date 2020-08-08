#/bin/bash
#
# Checks all tests in examples/cloc directory using make check. Programs return 0 for success or a number > 0 for failure.
#
echo ""
echo ""

if [ "$EPSDB" == "1" ] ; then
  echo Skipping cloc tests for now in EPSDB
  exit
fi

script_dir=$(dirname "$0")
pushd $script_dir
path=$(pwd)

echo ""
echo "RUNNING ALL TESTS IN: $path "
echo ""

echo "" > check-cloc.txt
echo "" >> check-cloc.txt
echo "Cloc Results:" >> check-cloc.txt
echo "*******A non-zero exit code means a failure occured.*******" >> check-cloc.txt
echo "***********************************************************" >> check-cloc.txt

#Loop over all directories and make run / make check depending on directory name
for directory in ./*/; do 
	(cd "$directory" && path=$(pwd) && base=$(basename $path) 
		make clean
		make
		if [ $? -ne 0 ]; then
			echo "$base: Make Failed" >> ../check-cloc.txt
		else
		  make run
		  echo " Return Code for $base: $?" >> ../check-cloc.txt
		fi
		make clean	
		
	)
	
done

#Print run.log for all tests that need visual inspection
for directory in ./*/; do
	(cd "$directory" && path=$(pwd) && base=$(basename $path)
	)
done
cat check-cloc.txt
popd

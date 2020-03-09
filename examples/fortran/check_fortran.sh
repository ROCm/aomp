#/bin/bash
#
# Checks all tests in examples/fortran directory using make check. Programs return 0 for success or a number > 0 for failure.
# Tests that need to be visually inspected: helloworld 
#
echo ""
echo ""
path=$(pwd)
base=$(basename $path)
echo ""
echo "RUNNING ALL TESTS IN: $path "
echo ""

echo "" > check-fortran.txt
echo "" >> check-fortran.txt
echo "Fortran Results:" >> check-fortran.txt
echo "*******A non-zero exit code means a failure occured.*******" >> check-fortran.txt
echo "*******Tests that need visual inspection: helloworld********" >> check-fortran.txt
echo "***********************************************************" >> check-fortran.txt

skiptests="bigloop"

#Loop over all directories and make run / make check depending on directory name
for directory in ./*/; do 
  (cd "$directory" && path=$(pwd) && base=$(basename $path)
    skip=0
    for test in $skiptests ; do
      if [ $test == $base ] ; then
        skip=1
        break
      fi
    done
    if [ $skip -ne 0 ] ; then
      echo "Skip $base!"
      continue
    fi
    make clean
    make
    make run
    echo " Return Code for $base: $?" >> ../check-fortran.txt
    make clean
  )
	
done

#Print run.log for all tests that need visual inspection
for directory in ./*/; do
	(cd "$directory" && path=$(pwd) && base=$(basename $path)
	)
done
cat check-fortran.txt

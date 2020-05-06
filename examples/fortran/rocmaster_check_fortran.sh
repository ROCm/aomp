#/bin/bash
#
# Checks all tests in examples/fortran directory using make check. Programs return 0 for success or a number > 0 for failure.
#

export AOMP=/opt/rocm/aomp

#Cleanup
rm -f check-fortran.txt
rm -f make-fail.txt
rm -f failing-tests.txt
rm -f passing-tests.txt

echo ""
echo ""
path=$(pwd)
base=$(basename $path)
echo ""
echo "RUNNING ALL TESTS IN: $path "
echo ""

echo "" > check-fortran.txt
echo "" >> check-fortran.txt
echo "*******A non-zero exit code means a failure occured.*******" >> check-fortran.txt
echo "***********************************************************" >> check-fortran.txt

skiptests=""

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
    else
		  make clean
		  make
      if [ $? -ne 0 ]; then
        echo " $base: Make Failed" >> ../make-fail.txt
      fi
		  make run
      result=$?
      if [ $result -ne 0 ]; then
		    echo " Return Code for $base: $result" >> ../failing-tests.txt
      else
		    echo " Return Code for $base: $result" >> ../passing-tests.txt
		  fi 
      echo " Return Code for $base: $result" >> ../check-fortran.txt
		fi
	)
	
done

#Print Return Codes
echo ""
if [ -e check-fortran.txt ]; then
  cat check-fortran.txt
fi
if [ -e make-fail.txt ]; then
  cat make-fail.txt
fi
echo ""

#Gather Test Data
if [ -e passing-tests.txt ]; then
  ((total_tests=$(wc -l <  passing-tests.txt)))
fi
if [ -e make-fail.txt ]; then
  ((total_tests+=$(wc -l <  make-fail.txt)))
fi
if [ -e failing-tests.txt ]; then
  ((total_tests+=$(wc -l <  failing-tests.txt)))
fi

#Print Results
echo -------------------- Results --------------------
echo "Number of tests: $total_tests"
echo ""
echo Passing tests: `wc -l <  passing-tests.txt`/$total_tests
echo ""

#Print failed tests
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

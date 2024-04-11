#!/bin/bash
#
# Runs examples in LIST
#

script_dir=$(dirname "$0")
pushd $script_dir
path=$(pwd)

function cleanup() {
  rm -f check-*.txt
  rm -f make-fail.txt
  rm -f failing-tests.txt
  rm -f passing-tests.txt
}
echo ""
echo -e "$ORG"RUNNING ALL TESTS IN: $path"$BLK"
echo ""

echo "************************************************************************************" 
echo "                   A non-zero exit code means a failure occured." 
echo "************************************************************************************"

#Loop over all directories and run the check script
if [ "$#" -ne 0 ]; then
  LIST="$@"
elif [ "$EPSDB" != "1" ]; then
  LIST="fortran hip openmp"
else
  LIST="fortran openmp"
fi
echo LIST: $LIST

for directory in $LIST; do
  pushd $directory > /dev/null
  cleanup
  echo "====== $directory ======" >> check-"$directory".txt
  for testdir in ./*/; do
    pushd $testdir > /dev/null
    testdir=$(echo $testdir | sed 's/.\///; s/\///')
		make clean
		make
    if [ $? -ne 0 ]; then
      echo "$testdir" >> ../make-fail.txt
      echo " Return Code for $testdir: Make Failed" >> ../check-"$directory".txt
    else
      make run
      result=$?
      if [ $result -ne 0 ]; then
        echo "$testdir" >> ../failing-tests.txt
      else
        echo "$testdir" >> ../passing-tests.txt
      fi
    echo " Return Code for $testdir: $result" >> ../check-"$directory".txt
    fi
    popd > /dev/null
  done
  echo "" >> check-"$directory".txt
  popd > /dev/null
done
echo -e "$ORG"FINAL RESULTS:"$BLK"
for directory in $LIST ; do
  (cd "$directory" && path=$(pwd) && base=$(basename $path)
    cat check-$base.txt
  )
done
popd

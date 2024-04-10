#!/bin/bash
#
# Runs examples in LIST
#

curdir=$PWD
realpath=`realpath $0`
realdir=$(dirname $realpath)
if [ $curdir != $realdir ] ; then
  for dir in `find $realdir -type d` ; do
    rdir=${dir#${realdir}/*}
    mkdir -p $rdir
  done
  _use_make_flag=1
  run_dir=$curdir
else
  _use_make_flag=0
  run_dir=$(dirname "$0")
fi

pushd $run_dir
path=$(pwd)

function cleanup() {
  rm -f check-*.txt
  rm -f make-fail.txt
  rm -f failing-tests.txt
  rm -f passing-tests.txt
}
echo ""
echo -e "$ORG"RUNNING ALL TESTS IN: $realdir"$BLK"
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
    _mf=""
    [ $_use_make_flag == 1 ] && _mf="-f $realdir/$directory/$testdir/Makefile"
		make $_mf clean
		make $_mf
    if [ $? -ne 0 ]; then
      echo "$testdir" >> ../make-fail.txt
      echo " Return Code for $testdir: Make Failed" >> ../check-"$directory".txt
    else
      make $_mf run
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

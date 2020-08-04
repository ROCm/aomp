#!/bin/bash
#
#  run_epsdb_aomp_test.sh
#

script_dir=$(dirname "$0")
parentdir=`eval "cd $script_dir;pwd;cd - > /dev/null"`
aompdir="$(dirname "$parentdir")"

set -x

# mygpu will eventually relocate to /opt/rocm/bin, support both cases for now.
if [ -a $AOMP/bin/mygpu ]; then
  export AOMP_GPU=`$AOMP/bin/mygpu`
  export EXTRA_OMP_FLAGS=--rocm-path=$AOMP/
else
  export AOMP_GPU=`$AOMP/../bin/mygpu`
fi

echo AOMP_GPU = $AOMP_GPU
$AOMP/bin/clang --version
ls /opt/rocm/amdgcn/bitcode
ls $AOMP/amdgcn/bitcode
rm -f  $aompdir/test/smoke/passing-tests.txt

set +x

cd $aompdir/test/smoke/helloworld
make clean
VERBOSE=1 make run

cd $aompdir/test/smoke
EPSDB=1 ./check_smoke.sh

echo $aompdir
echo

set -x
sort -f -d $aompdir/test/smoke/passing-tests.txt > $$ptests
sort -f -d $aompdir/bin/epsdb/epsdb_passes.txt > $$etests
cat $$etests
cat $$ptests
set +x

epasses=`diff $$etests $$ptests | grep '>' | wc -l`
echo Unexpected Passes $epasses
echo "====================="
diff $$etests $$ptests | grep '>' | sed 's/> //'
echo

efails=`diff $$etests $$ptests | grep '<' | wc -l`
echo Unexpected Fails $efails
echo "===================="
diff $$etests $$ptests | grep '<' | sed s'/< //'
echo
rm -f $$ptests $$etests
echo Done

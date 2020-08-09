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
mkdir -p ~/git/
cd $aompdir/bin
set +x
echo "======================"
./clone_aomp_test.sh

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
#cat $$etests
#cat $$ptests
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

echo "====== hip-openmp ==="
cd $aompdir/test/hip-openmp
AOMPHIP=$AOMP/.. ./check_hip-openmp.sh
echo "======= omp5 ==="
cd $aompdir/test/omp5
./check_omp5.sh  
echo "====== examples ==="
cd $aompdir/examples
EPSDB=1 AOMPHIP=$AOMP/.. ./check_examples.sh 

echo "======================"
./run_nekbone.sh
echo "======================"
cd ~/git/aomp-test/openmpapps
echo "======================"
./check_openmpapps.sh
# sollve take about 16 minutes
echo "======================"
./run_sollve.sh

echo Done



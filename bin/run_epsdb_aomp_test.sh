#!/bin/bash
# 
#  run_epsdb_aomp_test.sh
#

set -x

export AOMP_GPU=`$AOMP/bin/mygpu`
echo AOMP_GPU = $AOMP_GPU
export EXTRA_OMP_FLAGS=--rocm-path=$AOMP/
$AOMP/bin/clang --version
ls $AOMP/amdgcn/bitcode

set +x

cd ../test/smoke
EPSDB=1 ./check_smoke.sh 

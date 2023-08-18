#!/bin/bash

# --- Start standard header to set AOMP environment variables ----
realpath=`realpath $0`
thisdir=`dirname $realpath`
. $thisdir/aomp_common_vars
# --- end standard header ----

patchrepo $AOMP_REPOS_TEST/$AOMP_OMPTESTS_REPO_NAME

# Setup AOMP variables
 AOMP=${AOMP:-/usr/lib/aomp}

# Use function to set and test AOMP_GPU
setaompgpu

DEVICE_ARCH=${DEVICE_ARCH:-$AOMP_GPU}
DEVICE_TARGET=${DEVICE_TARGET:-amdgcn-amd-amdhsa}

echo DEVICE_ARCH   = $DEVICE_ARCH
echo DEVICE_TARGET = $DEVICE_TARGET

pushd $AOMP_REPOS_TEST/$AOMP_OMPTESTS_REPO_NAME
rm -f runtime-fails.txt
rm -f compile-fails.txt
rm -f passing-tests.txt
rm -f skipped-tests.txt
skip_list="t-tdpf-nested-parallel t-reduction-struct t-reduction-team t-unified-reduction-team t-reduction t-unified-reduction t-critical t-unified-critical t-unified-target-nowait-dep3 t-unified-target-nowait-dep2 t-unified-target-dep-2dev t-unified-target-depend-host-device t-unified-target-nowait-dep-implicit t-unified-parallel-target-nowait t-unified-target-nowait t-unified-array-of-ptr t-unified-class t-unified-declare-target t-unified-declare-target-lists t-unified-implicit-declare-target-no-dtor t-unified-runtime-calls t-unified-same-name-definitions t-unified-target-large-array t-unified-target-map-ptr t-unified-target-user-pinned t-unified-task t-unified-tp-nested-parallel t-unified-target-nowait-dep1 t-target-dep-2dev t-target-nowait-dep-implicit t-target-nowait-dep2 t-unified-target-nowait-dep1 t-target-depend-host-device t-target-nowait-dep1 t-target-nowait-dep3 t-unified-tp-nested-parallel t-unified-ttdpf-nested-parallel t-unified-ttdpfs-nested-parallel t-taskgroup"

# Add skip_list tests to runtime fails
for omp_test in $skip_list; do
  echo $omp_test > $AOMP_REPOS_TEST/$AOMP_OMPTESTS_REPO_NAME/skipped-tests.txt
done

# Move tests to avoid soft hang
if [ "$SKIP_TESTS" != 0 ]; then
  for omp_test in $skip_list; do
    if [ -d $omp_test ]; then
      mv $omp_test test-$omp_test-fail
    fi
  done
fi

log=$(date --iso-8601=minutes).log

echo env TARGET="-fopenmp-targets=$DEVICE_TARGET -fopenmp -Xopenmp-target=$DEVICE_TARGET -march=$DEVICE_ARCH" HOSTRTL=$AOMP/lib/libdevice TARGETRTL=$AOMP/lib GLOMPRTL=$AOMP/lib LLVMBIN=$AOMP/bin make -i

env TARGET="-fopenmp-targets=$DEVICE_TARGET -fopenmp -Xopenmp-target=$DEVICE_TARGET -march=$DEVICE_ARCH" HOSTRTL=$AOMP/lib/libdevice TARGETRTL=$AOMP/lib GLOMPRTL=$AOMP/lib LLVMBIN=$AOMP/bin make -i 2>&1 | tee omptests_run_$log

$thisdir/check_omptests.sh

removepatch $AOMP_REPOS_TEST/$AOMP_OMPTESTS_REPO_NAME

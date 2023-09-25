#!/bin/bash

# --- Start standard header to set AOMP environment variables ----
realpath=`realpath $0`
thisdir=`dirname $realpath`
export AOMP_USE_CCACHE=0

. $thisdir/aomp_common_vars
# --- end standard header ----

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
skip_list="t-tdpf-nested-parallel t-reduction-team t-unified-reduction-team t-reduction t-unified-reduction t-critical t-unified-target-nowait-dep3 t-unified-target-nowait-dep2 t-unified-target-dep-2dev t-unified-target-depend-host-device t-unified-target-nowait-dep-implicit t-unified-parallel-target-nowait t-unified-target-nowait t-unified-array-of-ptr t-unified-class t-unified-declare-target t-unified-declare-target-lists t-unified-implicit-declare-target-no-dtor t-unified-runtime-calls t-unified-same-name-definitions t-unified-target-large-array t-unified-target-map-ptr t-unified-target-user-pinned t-unified-task t-target-dep-2dev t-target-nowait-dep-implicit t-target-nowait-dep2 t-unified-target-nowait-dep1 t-target-depend-host-device t-target-nowait-dep1 t-target-nowait-dep3 t-unified-tp-nested-parallel t-unified-ttdpf-nested-parallel t-unified-ttdpfs-nested-parallel t-taskgroup"

# When HSA_XNACK is explicitely set to 0 by the user assume that they do not
# want to run the unified memory tests so add them to the skip list:
if [[ $HSA_XNACK == 0 ]]; then
  echo "Skipping all unified memory tests since HSA_XNACK is 0."
  skip_list="${skip_list} t-unified-critical t-unified-barrier t-unified-concurrent-target t-unified-data-sharing t-unified-data-sharing-many-teams t-unified-declare-simd t-unified-defaultmap t-unified-distribute t-unified-distribute-parallel-for-back2back t-unified-distribute-simd t-unified-distribute-simd-clauses t-unified-distribute-simd-dist-clauses t-unified-dpf t-unified-dpfs t-unified-dpfs-dist-clauses t-unified-exceptions t-unified-firstprivate-of-reference t-unified-flush t-unified-for t-unified-for-simd t-unified-ignore-unmappable-types t-unified-implicit-firstprivate t-unified-is-device-ptr-all-directives t-unified-l2-parallel t-unified-large-args t-unified-map-more-than t-unified-master t-unified-multiple-compilation-units t-unified-multiple-parallel t-unified-parallel t-unified-parallel-for t-unified-parallel-for-simd t-unified-parforsimd t-unified-partial-struct t-unified-reduction-struct t-unified-sections t-unified-sequence-distribute-parallel-for t-unified-shared-address-space t-unified-share-reference-orphan-directive t-unified-simd t-unified-single t-unified-target-api t-unified-target-basic t-unified-target-data-2map-same-array t-unified-target-enter-nowait t-unified-target-parallel t-unified-target-parallel-for t-unified-target-parallel-for-simd t-unified-target-parallel-for-simd-clauses t-unified-target-teams t-unified-target-teams-distribute t-unified-target-teams-distribute-parallel-for t-unified-target-teams-distribute-parallel-for-simd t-unified-target-teams-distribute-simd t-unified-target-update t-unified-target-update-not-there t-unified-target-update-nowait t-unified-teams-distribute t-unified-teams-distribute-parallel-for t-unified-teams-distribute-parallel-for-simd"
fi

# Add skip_list tests to runtime fails
for omp_test in $skip_list; do
  echo $omp_test > $AOMP_REPOS_TEST/$AOMP_OMPTESTS_REPO_NAME/skipped-tests.txt
done

# Move tests to avoid soft hang
if [ "$SKIP_TESTS" != 0 ]; then
  for omp_test in $skip_list; do
    if [ -d test-$omp_test-fail ]; then
      rm -rf test-$omp_test-fail
    fi
    if [ -d $omp_test ]; then
      mv $omp_test test-$omp_test-fail
    fi
  done
fi

log=$(date --iso-8601=minutes).log

echo env DEVICE_TYPE=amd DEVICE_TARGET=$DEVICE_TARGET DEVICE_ARCH=$DEVICE_ARCH HOSTRTL=$AOMP/lib/libdevice TARGETRTL=$AOMP/lib GLOMPRTL=$AOMP/lib LLVMBIN=$AOMP/bin make -i

env DEVICE_TYPE=amd DEVICE_TARGET=$DEVICE_TARGET DEVICE_ARCH=$DEVICE_ARCH HOSTRTL=$AOMP/lib/libdevice TARGETRTL=$AOMP/lib GLOMPRTL=$AOMP/lib LLVMBIN=$AOMP/bin make -i 2>&1 | tee omptests_run_$log

$thisdir/check_omptests.sh

# Move tests back to avoid polluting the repo
if [ "$SKIP_TESTS" != 0 ]; then
  for omp_test in $skip_list; do
    if [ -d $omp_test ]; then
      rm -rf $omp_test
    fi
    if [ -d test-$omp_test-fail ]; then
      mv test-$omp_test-fail $omp_test
    fi
  done
fi

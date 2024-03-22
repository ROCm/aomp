#!/bin/bash
save_status() { tval=$? && (($tval!=0)) && rval=$tval; }; rval=0
echo "ZZZZZZZZZZ BabelStream"   # marker for extract-tests
set -x
./milestone-3-babel -n 20               # 805.3 MB
save_status
./milestone-3-babel -n 20 -s 100000000  # 2400.0 MB
save_status
./milestone-3-babel -n 20 -s 200000000  # 4800.0 MB
save_status
if [[ "$AOMP_GPU" =~ ^gfx900 ]] ||
   [[ "$AOMP_GPU" =~ ^gfx906 ]] ||
   [[ "$AOMP_GPU" =~ ^gfx908 ]] ||
   [[ "$AOMP_GPU" =~ ^gfx10.* ]]; then
    echo "$AOMP_GPU: Skipping remaining larger memory tests"
    exit $rval
fi
./milestone-3-babel -n 10 -s 500000000  # 12000.0 MB
save_status
if [[ "$AOMP_GPU" =~ ^gfx11.* ]]; then
    echo "$AOMP_GPU: Skipping remaining larger memory tests"
    exit $rval
fi

# stop here for smoke testing so we don't exceed smoke test timeout
exit $rval

./milestone-3-babel -n 5  -s 1000000000 # 24000.0 MB
save_status
exit $rval

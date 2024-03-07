#!/bin/bash
echo "ZZZZZZZZZZ BabelStream"   # marker for extract-tests
set -x
./milestone-3-babel -n 20               # 805.3 MB
./milestone-3-babel -n 20 -s 100000000  # 2400.0 MB
./milestone-3-babel -n 20 -s 200000000  # 4800.0 MB
if [[ "$AOMP_GPU" =~ ^gfx10.* ]]; then
    echo "$AOMP_GPU: Skipping remaining larger memory tests"
    exit 0
fi
./milestone-3-babel -n 10 -s 500000000  # 12000.0 MB
if [[ "$AOMP_GPU" =~ ^gfx11.* ]]; then
    echo "$AOMP_GPU: Skipping remaining larger memory tests"
    exit 0
fi
./milestone-3-babel -n 5  -s 1000000000 # 24000.0 MB

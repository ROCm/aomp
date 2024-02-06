#!/bin/bash
echo "ZZZZZZZZZZ BabelStream"   # marker for extract-tests
set -x
./milestone-3-babel -n 20               # 805.3 MB
./milestone-3-babel -n 20 -s 100000000  # 2400.0 MB
./milestone-3-babel -n 20 -s 500000000  # 12000.0 MB
# skip 24GB, too big for smaller 16GB test devices
# ./milestone-3-babel -n 20 -s 1000000000 # 24000.0 MB

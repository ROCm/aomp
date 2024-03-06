#!/bin/bash
echo "ZZZZZZZZZZ BabelStream"   # marker for extract-tests
set -x
./milestone-3-babel-noteams -n 4                # 805.3 MB
./milestone-3-babel-noteams -n 4 -s 100000000   # 2400.0 MB
./milestone-3-babel-noteams -n 4 -s 200000000   # 4400.0 MB
# skip larger test cases, very slow without teams

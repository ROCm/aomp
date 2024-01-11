#!/bin/bash
set -x
# skip to avoid timeout (too slow without teams)
# ./milestone-3-babel-noteams -n 4 -s 1000000000
./milestone-3-babel-noteams -n 4 -s 100000000
./milestone-3-babel-noteams -n 4

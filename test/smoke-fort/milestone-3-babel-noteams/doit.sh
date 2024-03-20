#!/bin/bash
save_status() { tval=$? && (($tval!=0)) && rval=$tval; }; rval=0
echo "ZZZZZZZZZZ BabelStream"   # marker for extract-tests
set -x
./milestone-3-babel-noteams -n 4                # 805.3 MB
save_status
./milestone-3-babel-noteams -n 4 -s 100000000   # 2400.0 MB
save_status
# skip larger test cases, very slow without teams
exit $rval
./milestone-3-babel-noteams -n 4 -s 200000000   # 4400.0 MB
save_status

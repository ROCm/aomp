#!/bin/bash
save_status() { tval=$? && (($tval!=0)) && rval=$tval; }; rval=0
echo "ZZZZZZZZZZ BabelStream"   # marker for extract-tests
set -x
./milestone-3-babel -n 20               # 805.3 MB
save_status
exit $rval

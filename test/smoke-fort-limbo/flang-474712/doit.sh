#!/bin/bash
save_status() { tval=$? && (($tval!=0)) && rval=$tval; }; rval=0
echo "ZZZZZZZZZZ Jacobi"   # marker for extract-tests
set -x
./jacobi
save_status
exit $rval

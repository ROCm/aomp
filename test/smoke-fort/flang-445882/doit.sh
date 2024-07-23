#!/bin/bash
save_status() { tval=$? && (($tval!=0)) && rval=$tval; }; rval=0
echo "ZZZZZZZZZZ Derived Type Pointer"   # marker for extract-tests
set -x
$1
save_status
exit $rval

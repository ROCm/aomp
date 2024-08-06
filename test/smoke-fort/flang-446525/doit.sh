#!/bin/bash
save_status() { tval=$? && (($tval!=0)) && rval=$tval; }; rval=0
echo "ZZZZZZZZZZ rush_larsen_gpu_fomp"   # marker for extract-tests
set -x
./rush_larsen_gpu_omp_fort 100000 .00000001
save_status
./rush_larsen_gpu_omp_fort    100 10
save_status

# stop here for smoke testing so we don't exceed smoke test timeout
exit $rval

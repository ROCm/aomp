#!/bin/bash
set -x
rm -f ${TESTNAME}.log
./runtests.sh 2>&1 | tee ${TESTNAME}.log
touch ${TESTNAME}

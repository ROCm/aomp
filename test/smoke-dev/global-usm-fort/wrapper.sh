#!/bin/bash
./runtests.sh 2>&1 | tee ${TESTNAME}.log
touch ${TESTNAME}

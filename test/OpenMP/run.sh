#!/bin/bash

set -e

./build.sh

export HSA_IGNORE_SRAMECC_MISREPORT=1

for D in C CXX F F90; do
    cd $D
    make run
    cd ..
done


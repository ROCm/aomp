#!/bin/bash

if [[ $(wc -l $1 | awk '{print $1}') != 3 ]]; then
    exit 1
fi
grep -q 'Fortran STOP: Error message' $1

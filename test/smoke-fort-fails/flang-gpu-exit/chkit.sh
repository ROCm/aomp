#!/bin/bash

if [[ $(wc -l $1 | awk '{print $1}') != 1 ]]; then
    exit 1
fi
grep -q 'Launching kernel' $1

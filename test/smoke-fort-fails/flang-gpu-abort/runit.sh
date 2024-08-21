#!/bin/bash

{
    ./flang-gpu-abort
} 2>&1
if [[ $? != 0 ]]; then
    # turn whatever error code into error code 1
    exit 1
fi

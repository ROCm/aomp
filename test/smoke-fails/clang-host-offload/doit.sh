#!/bin/bash
export PATH=$AOMP/bin:$PATH
set -x
clang -g -O0 -fopenmp -fopenmp-targets=x86_64-unknown-linux-gnu -Xopenmp-target=x86_64-unknown-linux-gnu -march=znver1 -o offload offload.c
./offload
OMP_TARGET_OFFLOAD=mandatory ./offload

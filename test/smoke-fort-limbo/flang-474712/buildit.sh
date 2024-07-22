#!/bin/bash

export AOMP=${AOMP:-/opt/rocm/llvm}
export AOMP_GPU=${AOMP_GPU:-gfx90a}
export ROCM_GPU=${ROCM_GPU:-$AOMP_GPU}
export FLANG=${FLANG:-flang-new}
export FC=${FC:-$AOMP/bin/$FLANG}

make -f Makefile.jacobi clean
make -f Makefile.jacobi

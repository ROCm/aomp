#!/bin/bash

AOMP=${AOMP:-/opt/rocm/llvm}
AOMP_GPU=${AOMP_GPU:-gfx90a}
FLANG=${FLANG:-flang-new}
CLANG=${CLANG:-clang}
FC=${FC:-$AOMP/bin/$FLANG}
CC=${CC:-$AOMP/bin/$CLANG}
EXE=flang-target-associate-ptr

FFLAGS="-O3 -Werror -fopenmp --offload-arch=$AOMP_GPU"
CCLAGS="-O3 -Werror -fopenmp --offload-arch=$AOMP_GPU"
DEFINES="-DVERSION_STRING=4.0 -DUSE_OPENMPTARGET -DUSE_OMP_GET_WTIME"

set -x
rm -f $EXE *.o *.mod
$FC $DEFINES $FFLAGS -c device-c-omp.f90
$CC $DEFINES $CCLAGS -c device-omp.c
$FC $DEFINES $FFLAGS flang-target-associate-ptr.f90 device-c-omp.o device-omp.o -o $EXE

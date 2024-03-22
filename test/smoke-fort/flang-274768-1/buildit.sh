#!/bin/bash

AOMP=${AOMP:-/opt/rocm/llvm}
AOMP_GPU=${AOMP_GPU:-gfx90a}
FLANG=${FLANG:-flang-new}
FC=${FC:-$AOMP/bin/$FLANG}
EXE=a.out

FFLAGS="-O3 -Werror -fopenmp --offload-arch=$AOMP_GPU"

set -x
rm -f $EXE *.o *.mod
$FC $FFLAGS assumed-nosize.f90 |& tee comp.err
diff comp.err expected.err

#!/bin/bash

AOMP=${AOMP:-/opt/rocm/llvm}
AOMP_GPU=${AOMP_GPU:-gfx90a}
FLANG=${FLANG:-flang-new}
FC=${FC:-$AOMP/bin/$FLANG}
EXE=milestone-3-babel

FFLAGS="-O3 -Werror -fopenmp --offload-arch=$AOMP_GPU"
DEFINES="-DVERSION_STRING=4.0 -DUSE_OPENMPTARGET -DUSE_OMP_GET_WTIME -Dsimd="

set -x
rm -f $EXE *.o *.mod
$FC $DEFINES $FFLAGS -c BabelStreamTypes.F90
$FC $DEFINES $FFLAGS -c OpenMPTargetStream.F90
$FC $DEFINES $FFLAGS main.F90 OpenMPTargetStream.o BabelStreamTypes.o -o $EXE

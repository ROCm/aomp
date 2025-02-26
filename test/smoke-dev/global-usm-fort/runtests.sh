#!/bin/bash

AOMP=${AOMP:-/opt/rocm/llvm}
AOMP_GPU=${AOMP_GPU:-gfx90a}
FLANG=${FLANG:-flang-new}
FC=${FC:-$AOMP/bin/$FLANG}
CC=$AOMP/bin/clang
EXE=temp

FFLAGS="-Werror -fopenmp --offload-arch=$AOMP_GPU"
CFLAGS="-Werror -fopenmp --offload-arch=$AOMP_GPU"

set -x
rm -f $EXE *.o *.mod
$CC $CFLAGS -c helper.c
$FC $FFLAGS -c descr.f90

for r0 in "" -DDATA; do
for r1 in "" -DREQUIRES_USM; do
for r2 in "" -DDECLARE_TGT; do
  $FC $FFLAGS -o $EXE ${TESTNAME}.F90 $r0 $r1 $r2 -DXNACK=0 helper.o
  HSA_XNACK=0 ./$EXE
  $FC $FFLAGS -o $EXE ${TESTNAME}.F90 $r0 $r1 $r2 -DXNACK=1 helper.o
  HSA_XNACK=1 ./$EXE
done
done
done

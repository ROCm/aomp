#!/bin/bash
set -x

FC=$AOMP/bin/flang
CXX=$AOMP/bin/clang++
FFLAGS='-fopenmp --offload-arch=gfx90a'
$CXX -c timer.cpp -o timer.o
$CXX -c timer.f90 -o ftimer.o
$CXX -c allocator.f90 -o fallocator.o
$FC $FFLAGS reproducer.f90 -o a.out timer.o ftimer.o fallocator.o -lamdhip64 -lstdc++

OMP_TARGET_OFFLOAD=mandatory ./a.out

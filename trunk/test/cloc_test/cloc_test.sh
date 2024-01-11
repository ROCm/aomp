#!/bin/bash
#  File: cloc_test.sh

/bin/cat >set42.c <<"EOF"
#pragma omp declare target
extern void set42(int *r_p) { *r_p = 42; }
#pragma omp end declare target
EOF
TRUNK=${TRUNK:-$HOME/rocm/trunk}
FLANG=${FLANG:-flang-new}
echo $TRUNK/bin/clang -O3 -fopenmp --offload-arch=gfx908 -c -o set42.o set42.c
$TRUNK/bin/clang -O3 -fopenmp --offload-arch=gfx908 -c -o set42.o set42.c
echo $TRUNK/bin/$FLANG -O3 -fopenmp --offload-arch=gfx908 cloc_test.f95 set42.o -o cloc_test 
$TRUNK/bin/$FLANG -O3 -fopenmp --offload-arch=gfx908 cloc_test.f95 set42.o -o cloc_test 
echo ./cloc_test
./cloc_test

#include <stdio.h>
#include <omp.h>
#include <stdint.h>

#define TEST_ARRAY          1

/*
export "LD_LIBRARY_PATH=/gsa/yktgsa/home/e/i/eichen/new-tlomp/lomp/source/lib64:/usr/local/cuda/lib64"
export LIBRARY_PATH="/home/eichen/eichen/lnew/obj/lib"

/gsa/yktgsa/home/e/i/eichen/lnew/obj/bin/clang++ -v  -I/gsa/yktgsa/home/e/i/eichen/new-tlomp/lomp/source/lib64/ -I/gsa/yktgsa/home/e/i/eichen/new-tlomp/lomp/source/   -L/gsa/yktgsa/home/e/i/eichen/new-tlomp/lomp/source/lib64/ -L/gsa/yktgsa/home/e/i/eichen/new-tlomp/lomp/source/lib64/ -target powerpc64le-ibm-linux-gnu -mcpu=pwr8 -fopenmp=libomp -fopenmp-targets=nvptx64-nvidia-cuda -O3 test-array-inder.cpp
 */

#define N 640
#define C 64
#define P 10

int A[N];
int *p[P];

int main()
{
  // init
  int i;
  for(i=0; i<N; i++) A[i] = i;
  for(i=0; i<P; i++) p[i] = &A[i*C];

#if TEST_ARRAY
  #pragma omp target enter data map(to: A) map(alloc: p)
  for(i=0; i<P; i++) {
    //printf("%d: A 0x%lx, A[%3d] 0x%lx, p[i] 0x%lx -> 0x%lx\n", i, (unsigned long)&A[0], i*C, (unsigned long)&A[i*C], (unsigned long) &p[i], (unsigned long) p[i]);
    #pragma omp target enter data map(alloc: p[i][0:C])
  }

  #pragma omp target map(alloc: A, p) 
  {
    int i, j;
    for(i=0; i<P; i++) {
      //printf("%d: A 0x%lx, A[%3d] 0x%lx, p[i] 0x%lx -> 0x%lx\n", i, (unsigned long)&A[0], i*C, (unsigned long)&A[i*C], (unsigned long) &p[i], (unsigned long) p[i]);
    }
    for(i=0; i<P; i++) {
      for(j=0; j<C; j++) {
        p[i][j]++;
      } 
    }
  }

  #pragma omp target update from( A)

  int error = 0;
  for(i=0; i<N; i++) {
    if (A[i] != i+1) printf("%4d: got %d, expected %d, error %d\n", i, A[i], i+1, ++error);
  }
  printf("completed TEST ARRAY with %d errors\n", error);
#endif

  return 1;
}

#include <stdio.h>
#include <omp.h>
#include <stdint.h>


/*
export "LD_LIBRARY_PATH=/gsa/yktgsa/home/e/i/eichen/new-tlomp/lomp/source/lib64:/usr/local/cuda/lib64"
export LIBRARY_PATH="/home/eichen/eichen/lnew/obj/lib"

/gsa/yktgsa/home/e/i/eichen/lnew/obj/bin/clang++ -v  -I/gsa/yktgsa/home/e/i/eichen/new-tlomp/lomp/source/lib64/ -I/gsa/yktgsa/home/e/i/eichen/new-tlomp/lomp/source/   -L/gsa/yktgsa/home/e/i/eichen/new-tlomp/lomp/source/lib64/ -L/gsa/yktgsa/home/e/i/eichen/new-tlomp/lomp/source/lib64/ -target powerpc64le-ibm-linux-gnu -mcpu=pwr8 -fopenmp=libomp -fopenmp-targets=nvptx64-nvidia-cuda -O3 test-incl-array.cpp
 */

#define N 100
#define CHECK(_m, _i, _a, _b) if ((_a) != (_b)) printf("%d error for %s: got %d, expected %d, error %d\n", (_i), (_m), (_a), (_b), ++error)

int A[N];

struct S { int a, b, c, d; };

int main()
{
  int i, error=0;
  for(i=0; i<N; i++) A[i] = i;
  #pragma omp target enter data map(to: A[10:50])
  #pragma omp target //map(A[10:50])
  {
    for(int j=10; j<60; j++) A[j]++;
  }
  #pragma omp target exit data map(from: A[10:50])
  for(i=0; i<10; i++) CHECK("array test before mapped", i, A[i], i);
  for(i=10; i<60; i++) CHECK("array test mapped", i, A[i], i+1);
  for(i=60; i<N; i++) CHECK("array test after mapped", i, A[i], i);


  S s;
  s.a = 1; s.b = 2; s.c = 3; s.d = 4;
  #pragma omp target enter data map(to: s.b, s.c)
  #pragma omp target //map(A[10:50])
  {
    s.b++; s.c++;
  }
  #pragma omp target exit data map(from: s.b, s.c)
  CHECK("struct b", 0, s.b, 3);
  CHECK("struct c", 0, s.c, 4);

  printf("tests completed with %d errors\n", error);

  

  return 1;

}

/*
/gsa/yktgsa/home/e/i/eichen/lnew/obj/bin/clang -v -I/usr/local/cuda/include -I/gsa/yktgsa/home/e/i/eichen/new-tlomp/lomp/source/lib64/ -I/gsa/yktgsa/home/e/i/eichen/new-tlomp/lomp/source/   -L/gsa/yktgsa/home/e/i/eichen/new-tlomp/lomp/source/lib64/ -L/gsa/yktgsa/home/e/i/eichen/new-tlomp/lomp/source/lib64/ -fopenmp=libomp -O3  -target powerpc64le-ibm-linux-gnu -mcpu=pwr8 -fopenmp-targets=nvptx64-nvidia-cuda test-pinned.c -L /usr/local/cuda/lib64/ -lcudart

 */

#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include <assert.h>

#ifndef USE_PINNED
  // shoudl be set in the makefile, add here if tested directly
  #define USE_PINNED 1
#endif

#if USE_PINNED
  #include <cuda.h>
  #include <cuda_runtime.h>

  void *AllocMem(size_t memSize) {
    void *hostAddr;
    cudaError_t error;
  
    if (omp_get_num_devices() > 0) {
      fprintf(stderr, "used pinned mem\n");
      error = cudaMallocHost(&hostAddr, memSize);
      assert(error == cudaSuccess);
      return hostAddr;
    }
    fprintf(stderr, "used host mem\n");
    hostAddr = malloc(memSize);
    assert(hostAddr);
    return hostAddr;
  }
#else
  void *AllocMem(size_t memSize) {
    void *hostAddr;
  
    fprintf(stderr, "used malloc mem\n");
    hostAddr = malloc(memSize);
    assert(hostAddr);
    return hostAddr;
  }
#endif

#define N 1024

int *a, *b, *c;

int main() {
  int i, errors;

  // alloc
  a = (int *) AllocMem(N*sizeof(int));
  b = (int *) AllocMem(N*sizeof(int));
  c = (int *) AllocMem(N*sizeof(int));

  // init
  for (i=0; i<N; i++) {
    a[i] = i;
    b[i] = 2*i;
    c[i] = -1;
  }

  // test
  #pragma omp target map(to: a[0:N]) map(tofrom: b[0:N]) map(from: c[0:N])
  {
    for(int j=0; j<N; j++) {
      c[j] = a[j] + b[j];
      b[j]++;
    }
  }

  errors = 0;
  for(i=0; i<N; i++) {
    int bb = 2*i+1;
    if (bb != b[i]) printf("%d: b expected %d, got %d, error %d\n", i, bb, b[i], ++errors);
    if (errors>20) break;
  }

  for(i=0; i<N; i++) {
    int cc = 3*i;
    if (cc != c[i]) printf("%d: c expected %d, got %d, error %d\n", i, cc, c[i], ++errors);
    if (errors>20) break;
  }
  printf("got %d errors\n", errors);
  return 1;
}

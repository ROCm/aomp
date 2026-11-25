// MIT License
//
// Copyright (c) 2017 Advanced Micro Devices, Inc. All Rights Reserved.
//
// Permission is hereby granted, free of charge, to any person
// obtaining a copy of this software and associated documentation
// files (the "Software"), to deal in the Software without
// restriction, including without limitation the rights to use, copy,
// modify, merge, publish, distribute, sublicense, and/or sell copies
// of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be
// included in all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
// EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
// MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
// NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS
// BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN
// ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
// CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

// This program replicates OpenMP behavior for two (extremely reduced)
// kernels of the benchmark SPECaccel 2023 452.ep, emulating OpenMP's
// copy and zero-copy runtime behaviors.


#include <cstdlib>
#include <cstdio>
#include <sys/time.h>
#include <iostream>
#include <cmath>

#include "hip/hip_runtime.h"

__global__ void init_xx(double *xx, int length) {
  int i = threadIdx.x + blockIdx.x*blockDim.x;
  if (i > length) return;
  xx[i] = 1.0;   
}

__global__ void inc_xx(double *xx, int blksize, int nk) {
  int k = threadIdx.x + blockIdx.x*blockDim.x;
  if (k >= blksize) {
    return;
  }
  for(int i=0; i<2*nk; i++) {    
    xx[k*2*nk + i] += 1.0;
  }
  return;
}

int main() {
  int blksize = 15000;
  int nk = 65536;
  double *xx = (double *)malloc(blksize*2*nk*sizeof(double));
  int m = 40;
  int mk = 16;
  int mm = m - mk;
  int np = (1 << mm);
  int numblks = ceil( (double)np / (double) blksize);
  hipError_t err;
  
  printf("numblks = %d\n", numblks);

  char *HSA_XNACK_Env = getenv("HSA_XNACK");
  bool isXnackEnabled = false;
  if (HSA_XNACK_Env) {
    int HSA_XNACK_Val = atoi(HSA_XNACK_Env);
    isXnackEnabled = (HSA_XNACK_Val > 0) ? true : false;
  }
  
  double *d_xx = nullptr;
  //#pragma omp target enter data map(alloc:xx[0:blksize*2*nk])
  if (!isXnackEnabled) { // Copy
    printf("OpenMP Copy configuration\n");
    err = hipMalloc(&d_xx, blksize*2*nk*sizeof(double));
    if (err != HIP_SUCCESS) {
      printf("Cannot allocate device memory\n");
      return 0;
    }
    //hipMemcpy(d_xx, xx, blksize*2*nk*sizeof(double), hipMemcpyHostToDevice);
  } else {
    printf("OpenMP Zero-Copy configuration\n");
    d_xx = xx; // zero-copy
  }

  for (int blk=0; blk < 10; ++blk) {
    printf("blk=%d\n", blk);
    // #pragma omp target teams loop collapse(2)
    // for(int k=0; k<blksize; k++)
    //   for(int i=0; i<2*nk; i++)
    // 	xx[k*2*nk + i] = 1.0;
    init_xx<<<7680000, 256, 0>>>(d_xx, blksize*2*nk);
    hipDeviceSynchronize();
    // #pragma omp target teams loop
    // for (int k = 0; k < blksize; k++)
    //   for(int i=0; i<2*nk; i++)    
    // 	xx[k*2*nk + i] += 1.0;    
    inc_xx<<<938, 16, 0>>>(d_xx, blksize, nk);
    hipDeviceSynchronize();
  }

  return 0;
}

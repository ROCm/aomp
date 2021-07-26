#include "hip/hip_runtime.h"
#include <stdio.h>
__global__ void init_array(float *arr1){
  int i = hipBlockIdx_x;
  arr1[i]=(i)*2.0;
}

extern "C" int fortran_callable_init(float **a, int N) {
 hipLaunchKernelGGL((init_array), dim3(1), dim3(N), 0, 0, *a);
 return 0;
}

#include <stdio.h>
#include <hip/hip_runtime.h>

// this works, but not what is being tested
#if 0
__global__ void set_kernel(int *a, int n) {
  for(int i = 0; i < n; i++)
    a[i] = 0;
}
#endif

void set_mem(int *a, int n) {
  hipError_t err;

  // this does not currently work
  err = hipMemset(a, 0, n * sizeof(int));
  if(err != HIP_SUCCESS)
    printf("Error %s!\n", hipGetErrorName(err));

  // this works
  //   set_kernel<<<1, 1, 0>>>(a, n);

  err = hipDeviceSynchronize();
  if(err != HIP_SUCCESS)
    printf("Error %s!\n", hipGetErrorName(err));
}

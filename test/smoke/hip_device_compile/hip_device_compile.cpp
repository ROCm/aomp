#include "hip/hip_runtime.h"
#include <stdio.h>
//no need for value verification, compile test only
__device__
void g() {}

__host__ __device__
void f()
{
#ifdef __HIP_DEVICE_COMPILE__
  g();
#endif
}

__global__
void kernel()
{
  f();
}

int main()
{
  kernel<<<1,1>>>();
  hipDeviceSynchronize();
  printf("Success!\n");
  return 0;
}

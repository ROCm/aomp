#include <hip/hip_runtime.h>
#include <stdlib.h>

__global__ void bar_kernel(int *x, int *y, int *z)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
 
    if (i < 1000) {
          z[i] += x[i] + y[i];
    }
}

extern "C" {

void bar_GPU(int *x, int *y, int *z)
{
    hipLaunchKernelGGL(bar_kernel, dim3(32), dim3(32), 0, 0, x, y, z);
}

} /* extern "C" */

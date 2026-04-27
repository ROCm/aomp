#include <hip/hip_runtime.h>
#include <stdlib.h>

__global__ void bar_kernel(int *x, int *y, int *z, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < n) {
          z[i] += x[i] + y[i];
    }
}

extern "C" {

void bar_GPU(int *x, int *y, int *z, int n)
{
    int num_threads = 256;
    int num_blocks = n / num_threads + 1;
    hipLaunchKernelGGL(bar_kernel, dim3(num_blocks), dim3(num_threads), 0, 0, x, y, z, n);
    (void)hipDeviceSynchronize();
}

} /* extern "C" */

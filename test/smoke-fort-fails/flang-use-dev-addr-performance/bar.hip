#include <hip/hip_runtime.h>
#include <stdlib.h>

__global__ void bar_kernel(int *x, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < n) {
           x[i]++;
    }
}

extern "C" {

void bar_GPU(int *x, int n)
{
    int num_threads = 256;
    int num_blocks = n / num_threads + 1;
    hipLaunchKernelGGL(bar_kernel, dim3(num_blocks), dim3(num_threads), 0, 0, x, n);
    (void)hipDeviceSynchronize();
}

void print_ptr(int * x) {
    static int cnt;
    printf("pointer x %lx number of calls: %d\n",(ulong)x, ++cnt);
}
} /* extern "C" */

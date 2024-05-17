#include <hip/hip_runtime.h>
__global__ void saxpy_kernel(size_t n, float * x, float * y) {
    size_t i = threadIdx.x + blockIdx.x * blockDim.x;
    y[i] = 2*x[i] + y[i];
}
extern "C" {
    void saxpy_hip(size_t n, float * x, float * y) {
        assert(n % 32 == 0);
        saxpy_kernel<<<n/32,32,0,NULL>>>(n, x, y);
    }
}

#include <hip/hip_runtime.h>

__global__ void add_kernel(int *a) {
    time_t start = clock64();
    while (clock64() - start < 1e9) {
    }
    a[0] += 2.0;
}

void launch_hip_add(hipStream_t *stream, int *a) {
    printf("Calling HIP kernel\n");
    add_kernel<<<1, 1, 0, *stream>>>(a);

}

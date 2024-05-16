#include <stdio.h>
#include <iostream>
#include "hip/hip_runtime.h"
#include <stdlib.h>
#include <time.h>

#define CHECK(cmd)                                                                                 \
    {                                                                                              \
        hipError_t error = cmd;                                                                    \
        if (error != hipSuccess) {                                                                 \
            fprintf(stderr, "error: '%s'(%d) at %s:%d\n", hipGetErrorString(error), error,         \
                    __FILE__, __LINE__);                                                           \
            exit(EXIT_FAILURE);                                                                    \
        }                                                                                          \
    }

__global__ void bit_extract_kernel(uint32_t* C_d, const uint32_t* A_d, size_t N) {
    size_t offset = (hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x);
    size_t stride = hipBlockDim_x * hipGridDim_x;

        for (size_t i = offset; i < N; i += stride) {
#ifdef __HIP_PLATFORM_HCC__
        C_d[i] = i;
#else /* defined __HIP_PLATFORM_NVCC__ or other path */
        C_d[i] = i;
#endif
    }
}

int main(int argc, char* argv[]) {
    uint32_t *A_d, *C_d;
    uint32_t *A_h, *C_h;
    struct timespec t0,t1,t2,t3,t4,t5;

    CHECK(hipInit( 0 ));
    int deviceId;
    CHECK(hipGetDevice(&deviceId));
    hipDeviceProp_t props;
    CHECK(hipGetDeviceProperties(&props, deviceId));
    fprintf(stderr, "running on device #%d %s\n", deviceId, props.name);
    fprintf(stderr, "Device num %s\n", props.gcnArchName);
    size_t N = 1000000000;
#if gfx900_supported
    // if we are a dinky memory us 2GB max.
    if (props.gcnArch < 906)
      N = N /2;
#endif
    size_t Nbytes = N * sizeof(uint32_t);

    CHECK(hipDeviceSynchronize());

    A_h = (uint32_t*)malloc(Nbytes);
    CHECK(A_h == 0 ? hipErrorMemoryAllocation : hipSuccess);
    C_h = (uint32_t*)malloc(Nbytes);
    CHECK(C_h == 0 ? hipErrorMemoryAllocation : hipSuccess);

    for (size_t i = 0; i < N; i++) {
        A_h[i] = i;
    }

    CHECK(hipMalloc(&A_d, Nbytes));
    CHECK(hipMalloc(&C_d, Nbytes));

    fprintf(stderr, "info: copy Host2Device\n");
    clock_gettime(CLOCK_REALTIME, &t0);
    CHECK(hipMemcpy(A_d, A_h, Nbytes, hipMemcpyHostToDevice));
    clock_gettime(CLOCK_REALTIME, &t1);
    double  m = (t1.tv_sec - t0.tv_sec) + (t1.tv_nsec - t0.tv_nsec)/1e9;
    fprintf(stderr, "Time %f for copy to device %ld bytes\n", m, N*4);
    fprintf(stderr, "%f GBytes/sec\n",  N*4/m/(1024*1024*1024));

    fprintf(stderr, "info: launch 'bit_extract_kernel' \n");
    const unsigned blocks = 512;
    const unsigned threadsPerBlock = 256;
    hipLaunchKernelGGL(bit_extract_kernel, dim3(blocks), dim3(threadsPerBlock), 0, 0, C_d, A_d, N);

    fprintf(stderr, "info: copy Device2Host\n");
    clock_gettime(CLOCK_REALTIME, &t0);
    CHECK(hipMemcpy(C_h, C_d, Nbytes, hipMemcpyDeviceToHost));
    clock_gettime(CLOCK_REALTIME, &t1);
    double  n = (t1.tv_sec - t0.tv_sec) + (t1.tv_nsec - t0.tv_nsec)/1e9;
    fprintf(stderr, "Time %f for copy to device %ld bytes\n", n, N*4);
    fprintf(stderr, "%f GBytes/sec\n",  N*4/n/(1024*1024*1024));

    fprintf(stderr, "PASSED !\n");
    CHECK(hipFree(A_d));
    CHECK(hipFree(C_d));
    free(A_h);
    free(C_h);
    return 0;
}

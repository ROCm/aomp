#include <stdio.h>
#include <iostream>
#include "hip/hip_runtime.h"
#include <thread>

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
        C_d[i] = __bitextract_u32(A_d[i], 8, 4);
#else /* defined __HIP_PLATFORM_NVCC__ or other path */
        C_d[i] = ((A_d[i] & 0xf00) >> 8);
#endif
    }
}

void* wrapper(void * start) {
    uint32_t *A_d, *C_d;
    uint32_t *A_h, *C_h;
    size_t N = 1000000;
    size_t Nbytes = N * sizeof(uint32_t);

    int mytid = (long)start;
    fprintf(stderr, "info: %d synchronize\n", mytid);
    CHECK(hipDeviceSynchronize());

    fprintf(stderr, "info: %d  allocate host mem (%6.2f MB)\n", mytid, 2 * Nbytes / 1024.0 / 1024.0);
    A_h = (uint32_t*)malloc(Nbytes);
    CHECK(A_h == 0 ? hipErrorMemoryAllocation : hipSuccess);
    C_h = (uint32_t*)malloc(Nbytes);
    CHECK(C_h == 0 ? hipErrorMemoryAllocation : hipSuccess);

    for (size_t i = 0; i < N; i++) {
        A_h[i] = i;
    }

    fprintf(stderr, "info: %d  allocate device mem (%6.2f MB)\n", mytid, 2 * Nbytes / 1024.0 / 1024.0);
    CHECK(hipMalloc(&A_d, Nbytes));
    CHECK(hipMalloc(&C_d, Nbytes));

    fprintf(stderr, "info: %d  allocate A:%lx C:%lx\n", mytid, (long)A_d, (long)C_d);

    fprintf(stderr, "info: %d  copy Host2Device\n", mytid);
    CHECK(hipMemcpy(A_d, A_h, Nbytes, hipMemcpyHostToDevice));

    fprintf(stderr, "info: %d  launch 'bit_extract_kernel' \n", mytid);
    const unsigned blocks = 512;
    const unsigned threadsPerBlock = 256;
    hipLaunchKernelGGL(bit_extract_kernel, dim3(blocks), dim3(threadsPerBlock), 0, 0, C_d, A_d, N);

    fprintf(stderr, "info: %d  copy Device2Host %lx\n", mytid, (long)C_d);
    CHECK(hipMemcpy(C_h, C_d, Nbytes, hipMemcpyDeviceToHost));

    fprintf(stderr, "info: %d  check result\n", mytid);
    for (size_t i = 0; i < N; i++) {
        unsigned Agold = ((A_h[i] & 0xf00) >> 8);
        if (C_h[i] != Agold) {
            fprintf(stderr, "mismatch detected.\n");
            printf("%zu: %08x =? %08x (Ain=%08x)\n", i, C_h[i], Agold, A_h[i]);
            CHECK(hipErrorUnknown);
        }
    }
    fprintf(stderr, "PASSED %d!\n", mytid);
    fprintf(stderr, "info: %d  free A:%lx C:%lx\n", mytid, (long)A_d, (long)C_d);
    CHECK(hipFree(A_d));
    CHECK(hipFree(C_d));
    free(A_h);
    free(C_h);
    return NULL;
}

long inc=0;
void doit() {
	wrapper((long*)++inc);
}

int main(int argc, char* argv[]) {
  CHECK(hipInit( 0 ));
  int deviceId;
  CHECK(hipGetDevice(&deviceId));
  hipDeviceProp_t props;
  CHECK(hipGetDeviceProperties(&props, deviceId));
  fprintf(stderr, "running on device #%d %s\n", deviceId, props.name);
  int nThreads = 100;
  if (argc > 1) nThreads = atoi(argv[1]);
  fprintf(stderr, "using %d threads\n",nThreads);

  std::thread tID[nThreads];

  for(long i=0;i<nThreads;i++)
    tID[i]=std::thread(doit);
  for(long i=0;i<nThreads;i++)
	tID[i].join();
  return 0;
}


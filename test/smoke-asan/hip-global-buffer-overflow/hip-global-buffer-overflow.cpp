#define __HIP_PLATFORM_AMD__ 1
#include <hip/hip_runtime.h>

#define N 100

void printHipError(hipError_t error) {
  printf("Hip Error: %s\n", hipGetErrorString(error));
}

bool hipCallSuccessfull(hipError_t error) {
  if (error != hipSuccess)
    printHipError(error);
  return error == hipSuccess;
}

__device__ int D_Ptr[N];

__global__ void Initialize(int n) {
  int index = blockDim.x * blockIdx.x + threadIdx.x;
  if (index < n) {
    D_Ptr[index + 1] = 2 * (index + 1);
  }
}

int main(int argc, char *argv[]) {
  size_t NBytes = N * sizeof(int);
  int NumOfThreadBlocks = (N + 64 - 1) / 64;
  int ThreadBlockSize = 64;
  hipLaunchKernelGGL(Initialize, dim3(NumOfThreadBlocks), dim3(ThreadBlockSize),
                     0, 0, N);
  return 0;
}

/// CHECK:=================================================================
/// CHECK-NEXT:=={{[0-9]+}}==ERROR: AddressSanitizer: global-buffer-overflow on amdgpu device 0 at pc [[PC:.*]]
/// CHECK-NEXT:WRITE of size 4 in workgroup id ({{[0-9]+}},0,0)
/// CHECK-NEXT:  #0 [[PC]] in Initialize(int) at {{.*}}aomp/test/smoke-asan/hip-global-buffer-overflow/hip-global-buffer-overflow.cpp:21:{{[0-9]+}}

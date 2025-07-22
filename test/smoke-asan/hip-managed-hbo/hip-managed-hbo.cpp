#define __HIP_PLATFORM_AMD__ 1
#include <hip/hip_runtime.h>

void printHipError(hipError_t error) {
  printf("Hip Error: %s\n", hipGetErrorString(error));
}

bool hipCallSuccessfull(hipError_t error) {
  if (error != hipSuccess)
    printHipError(error);
  return error == hipSuccess;
}

__global__ void TestKernel(int n, int *ptr) {
  int index = blockDim.x * blockIdx.x + threadIdx.x;
  if (index < n) {
    ptr[index + 1] = 2 * (index + 1);
  }
}

int main(int argc, char *argv[]) {
  int N = 100;
  size_t NBytes = N * sizeof(int);
  int *D_Ptr;
  int NumOfThreadBlocks = (N + 64 - 1) / 64;
  int ThreadBlockSize = 64;
  hipCallSuccessfull(hipMallocManaged(&D_Ptr, NBytes));
  hipLaunchKernelGGL(TestKernel, dim3(NumOfThreadBlocks), dim3(ThreadBlockSize),
                     0, 0, N, D_Ptr);
  hipCallSuccessfull(hipFree(D_Ptr));
  return 0;
}

/// CHECK:=================================================================
/// CHECK-NEXT:=={{[0-9]+}}==ERROR: AddressSanitizer: heap-buffer-overflow on amdgpu device 0 at pc [[PC:.*]]
/// CHECK-NEXT:WRITE of size 4 in workgroup id ({{[0-9]+}},0,0)
/// CHECK-NEXT:  #0 [[PC]] in TestKernel(int, int*) at {{.*}}aomp/test/smoke-asan/hip-managed-hbo/hip-managed-hbo.cpp:17:{{[0-9]+}}

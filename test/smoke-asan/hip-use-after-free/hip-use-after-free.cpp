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

__global__ void Initialize(int n, int *ptr) {
  int index = blockDim.x * blockIdx.x + threadIdx.x;
  if (index < n) {
    ptr[index] = 2 * (index + 1);
  }
}

int main(int argc, char *argv[]) {
  int N = 100;
  size_t NBytes = N * sizeof(int);
  int *H_Ptr = new int[N];
  int *D_Ptr;
  int NumOfThreadBlocks = (N + 64 - 1) / 64;
  int ThreadBlockSize = 64;
  hipCallSuccessfull(hipMalloc(&D_Ptr, NBytes));
  hipCallSuccessfull(hipFree(D_Ptr));
  hipLaunchKernelGGL(Initialize, dim3(NumOfThreadBlocks), dim3(ThreadBlockSize),
                     0, 0, N, D_Ptr);
  hipCallSuccessfull(hipMemcpy(H_Ptr, D_Ptr, NBytes, hipMemcpyDeviceToHost));
  delete[] H_Ptr;
  return 0;
}

/// CHECK:=================================================================
/// CHECK-NEXT:=={{[0-9]+}}==ERROR: AddressSanitizer: heap-use-after-free on amdgpu device 0 at pc [[PC:.*]]
/// CHECK-NEXT:WRITE of size 4 in workgroup id ({{[0-9]+}},0,0)
/// CHECK-NEXT:  #0 [[PC]] in Initialize(int, int*) at {{.*}}aomp/test/smoke-asan/hip-use-after-free/hip-use-after-free.cpp:17:11

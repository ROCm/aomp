#define __HIP_PLATFORM_AMD__ 1
#include <hip/hip_runtime.h>

void printHipError(hipError_t error) {
  printf("Hip Error: %s\n", hipGetErrorString(error));
}

bool hipCallSuccessfull(hipError_t error) {
  if (error != hipSuccess) printHipError(error);
  return error == hipSuccess;
}

__global__ void TestKernel(int n) {
  int *dptr = (int *)malloc(n*sizeof(int));
  size_t iX = blockDim.x * blockIdx.x + threadIdx.x;
  free(dptr);
  dptr[iX] = 2 * (iX + 1);
}

int main(int argc, char *argv[]) {
  int N = 1;
  size_t NBytes = N * sizeof(int);
  int *H_Ptr = new int[N];
  int *D_Ptr;
  int NumOfThreadBlocks = (N + 64 - 1) / 64;
  int ThreadBlockSize = 64;
  hipCallSuccessfull(hipMalloc(&D_Ptr, NBytes));
  hipLaunchKernelGGL(TestKernel, dim3(NumOfThreadBlocks), dim3(ThreadBlockSize), 0, 0, N);
  hipCallSuccessfull(hipMemcpy(H_Ptr, D_Ptr, NBytes, hipMemcpyDeviceToHost));
  hipCallSuccessfull(hipFree(D_Ptr));
  delete[] H_Ptr;
  return 0;
}

/// CHECK:=================================================================
/// CHECK-NEXT:=={{[0-9]+}}==ERROR: AddressSanitizer: heap-use-after-free on amdgpu device 0 at pc [[PC:0x[0-9a-fA-F]+]]
/// CHECK-NEXT:WRITE of size 4 in workgroup id ({{[0-9]+}},0,0)
/// CHECK-NEXT:  #0 [[PC]] in TestKernel(int) at {{.*}}aomp/test/smoke-asan/hip-device-free-uaf/hip-device-free-uaf.cpp:17:{{[0-9]+}}
/// CHECK:{{0x[0-9a-fA-F]+}} is 0 bytes above an address from a device malloc (or free) call of size 4 from
/// CHECK-NEXT:  #0 {{0x[0-9a-fA-F]+}} in TestKernel(int) at {{.*}}aomp/test/smoke-asan/hip-device-free-uaf/hip-device-free-uaf.cpp:16:{{[0-9]+}}

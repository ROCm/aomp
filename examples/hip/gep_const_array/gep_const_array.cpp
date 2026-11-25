#include "hip/hip_runtime.h"
#include <stdio.h>

void printHipError(hipError_t error) {
  printf("Hip Error: %s\n", hipGetErrorString(error));
}

bool hipCallSuccessful(hipError_t error) {
  if (error != hipSuccess)
    printHipError(error);
  return error == hipSuccess;
}

bool deviceCanCompute(int deviceID) {
  bool canCompute = false;
  hipDeviceProp_t deviceProp;
  bool devicePropIsAvailable =
      hipCallSuccessful(hipGetDeviceProperties(&deviceProp, deviceID));
  if (devicePropIsAvailable) {
    canCompute = deviceProp.computeMode != hipComputeModeProhibited;
    if (!canCompute)
      printf("Compute mode is prohibited\n");
  }
  return canCompute;
}

bool deviceIsAvailable(int *deviceID) {
  return hipCallSuccessful(hipGetDevice(deviceID));
}

// We always use device 0
bool haveComputeDevice() {
  int deviceID = 0;
  return deviceIsAvailable(&deviceID) && deviceCanCompute(deviceID);
}

#define N 100
#define NTHS 1024

__global__ void accessConstArray(int idx, int *ret) {
  __shared__ int const_arr[NTHS];

  const_arr[threadIdx.x] = threadIdx.x;
  ret[threadIdx.x] = const_arr[threadIdx.x];
}

int main() {
  int hRet[NTHS];
  int *dRet = nullptr;
  int idx = 10;

  if (!haveComputeDevice()) {
    printf("No compute device available\n");
    return 1;
  }

  bool retAllocated = hipCallSuccessful(
    hipMalloc((void **)&dRet, NTHS*sizeof(int)));

  if (!retAllocated) {
    printf("Error allocating device memory\n");
    return 1;
  }

  accessConstArray<<<1, NTHS, 0, 0>>>(idx, dRet);

  bool copyBack = hipCallSuccessful(
    hipMemcpy(&hRet, dRet, NTHS*sizeof(int), hipMemcpyDeviceToHost));
  if (!copyBack) {
    printf("Unable to copy memory from device to host\n");
    return 1;
  }

  for(int i = 0; i < NTHS; i++)
    if (hRet[i] != i) {
      printf("Error at %d\n", i);
      return 1;
    }

  hipFree(dRet);

  return 0;
}

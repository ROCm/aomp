// MIT License
//
// Copyright (c) 2017 Advanced Micro Devices, Inc. All Rights Reserved.
//
// Permission is hereby granted, free of charge, to any person
// obtaining a copy of this software and associated documentation
// files (the "Software"), to deal in the Software without
// restriction, including without limitation the rights to use, copy,
// modify, merge, publish, distribute, sublicense, and/or sell copies
// of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be
// included in all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
// EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
// MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
// NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS
// BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN
// ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
// CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

#include "hip/hip_runtime.h"
#include <cstdio>
#include <cstdlib>
#include <cstring>

#define N 100

__device__ int D_A[N], D_B[N], D_C[N];

__global__ void initialize(int n) {
  size_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    D_A[i] = 2 * (i + 1);
    D_B[i] = 3 * (i + 1);
  }
}

__global__ void vecadd(int n) {
  size_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    D_C[i + 100] = D_A[i] + D_B[i];
  }
}

__global__ void evaluateResult(int n, bool *flag) {
  size_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    if (D_C[i] != D_A[i] + D_B[i])
      flag[i] = false;
    if (D_C[i] == D_A[i] + D_B[i])
      flag[i] = true;
  }
}

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

void printArray(int n, int *arr) {
  for (int i = 0; i < n; i++) {
    printf("\n%d", arr[i]);
  }
}

void checkResult(int n, bool *flag) {
  for (int i = 0; i < n; i++) {
    if (!flag[i]) {
      printf("\nError!!\n");
      break;
    }
    if (i + 1 == n)
      printf("\nSuccess!!\n");
  }
}

int main(int argc, char *argv[]) {

  if (!haveComputeDevice()) {
    printf("No compute device available\n");
    return 0;
  }

  bool flag[N];
  memset(flag, false, N * sizeof(bool));
  bool *D_flag;

  hipCallSuccessful(hipMalloc(&D_flag, sizeof(bool) * N));
  hipCallSuccessful(
      hipMemcpy(D_flag, flag, sizeof(bool) * N, hipMemcpyHostToDevice));

  hipLaunchKernelGGL(initialize, dim3((N + 64 - 1) / 64, 1, 1), dim3(64, 1, 1),
                     0, 0, N);
  hipLaunchKernelGGL(vecadd, dim3((N + 64 - 1) / 64, 1, 1), dim3(64, 1, 1), 0,
                     0, N);
  hipLaunchKernelGGL(evaluateResult, dim3((N + 64 - 1) / 64, 1, 1),
                     dim3(64, 1, 1), 0, 0, N, D_flag);

  hipDeviceSynchronize();

  hipCallSuccessful(
      hipMemcpy(flag, D_flag, sizeof(bool) * N, hipMemcpyDeviceToHost));

  checkResult(N, D_flag);

  hipFree(D_flag);

  return 0;
}

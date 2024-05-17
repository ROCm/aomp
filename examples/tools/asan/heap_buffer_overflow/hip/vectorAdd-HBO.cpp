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

#define N 100

__global__ void vecadd(int n, int *A, int *B, int *C) {
  size_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    C[i] = A[i] + B[i];
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

void initialize(int n, int *arr) {
  srand(time(0));
  for (int i = 0; i < n; i++) {
    arr[i] = rand() % N;
  }
}

void checkResult(int n, int *C, int *A, int *B) {
  bool flag = false;
  for (int i = 0; i < n; i++) {
    if (C[i] == A[i] + B[i])
      continue;
    flag = true;
    break;
  }
  if (!flag)
    printf("\nSuccess!!\n");
  else
    printf("\nError!!\n");
}

int main(int argc, char *argv[]) {

  size_t NBytes = N * sizeof(int);

  int *h_A = (int *)malloc(NBytes);
  int *h_B = (int *)malloc(NBytes);
  int *h_C = (int *)malloc(NBytes);

  initialize(N, h_A);
  initialize(N, h_B);

  if (!haveComputeDevice()) {
    printf("No compute device available\n");
    return 0;
  }

  int *d_A, *d_B, *d_C;

  hipCallSuccessful(hipMalloc(&d_A, NBytes));
  hipCallSuccessful(hipMalloc(&d_B, NBytes));
  hipCallSuccessful(hipMalloc(&d_C, NBytes));

  hipCallSuccessful(hipMemcpy(d_A, h_A, NBytes, hipMemcpyHostToDevice));
  hipCallSuccessful(hipMemcpy(d_B, h_A, NBytes, hipMemcpyHostToDevice));

  hipLaunchKernelGGL(vecadd, dim3((N + 64 - 1) / 64, 1, 1), dim3(64, 1, 1), 0,
                     0, N, d_A, d_B, d_C);

  hipCallSuccessful(hipMemcpy(h_C, d_C, NBytes, hipMemcpyDeviceToHost));

  checkResult(N, h_C, h_A, h_B);
  hipFree(d_A);
  hipFree(d_B);
  hipFree(d_C);
  free(h_A);
  free(h_B);
  free(h_C);

  return 0;
}

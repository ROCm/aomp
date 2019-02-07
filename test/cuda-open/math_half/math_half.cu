// MIT License
//
// Copyright (c) 2018 Advanced Micro Devices, Inc. All Rights Reserved.
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

// These test only check if the code compiles, we don't test
// functionality yet.
// Reference: Cuda Toolkit v 9.2.88
//  1.1 Half Precision Intrinsics

#include <stdio.h>
#include <hip/hip_host_runtime_api.h>
#include <cuda_fp16.h>
#define N 10

__global__
void testHalfMath(__half *b)
{
  int i = blockIdx.x;
  unsigned u = (unsigned) i;
  __half h = (__half) i;
  if (i<N) {
    // 1.1 Half Precision Intrinsics

    // 1.1.1 Half Arithmetic Functions
    // b[i] = __hadd(h,h); // Fixme: missing functions __nv_hadd
    // b[i] = __hadd_sat(h,h); // Fixme: missing functions __nv_hadd_sat
    //  b[i] = __hdiv(h,h); // Fixme: Add __hdiv to cuda_open_headers
    b[i] = __hadd(h, h);
  }
}

void printArray(__half *array)
{
  printf("[");
  bool first = true;
  for (int i = 0; i<N; ++i)
  {
    if (first)
    {
      printf("%d", (int)array[i]);
      first = false;
    }
    else
    {
      printf(", %d", (int)array[i]);
    }
  }
  printf("]");
}

void printHipError(hipError_t error)
{
  printf("Hip Error: %s\n", hipGetErrorString(error));
}

bool hipCallSuccessful(hipError_t error)
{
  if (error != hipSuccess)
    printHipError(error);
  return error == hipSuccess;
}

bool deviceCanCompute(int deviceID)
{
  bool canCompute = false;
  hipDeviceProp_t deviceProp;
  bool devicePropIsAvailable =
    hipCallSuccessful(hipGetDeviceProperties(&deviceProp, deviceID));
  if (devicePropIsAvailable)
  {
    canCompute = deviceProp.computeMode != hipComputeModeProhibited;
    if (!canCompute)
      printf("Compute mode is prohibited\n");
  }
  return canCompute;
}

bool deviceIsAvailable(int *deviceID)
{
  return hipCallSuccessful(hipGetDevice(deviceID));
}

// We always use device 0
bool haveComputeDevice()
{
  int deviceID = 0;
  return deviceIsAvailable(&deviceID) && deviceCanCompute(deviceID);
}

int main()
{

  __half hostArray[N];

  if (!haveComputeDevice())
  {
    printf("No compute device available\n");
    return 0;
  }

  for (int i = 0; i<N; ++i)
    hostArray[i] = 0;

  printf("Array content before kernel:\n");
  printArray(hostArray);
  printf("\n");

  __half *deviceArray;
  if (!hipCallSuccessful(hipMalloc((void **)&deviceArray, N*sizeof(int))))
  {
    printf("Unable to allocate device memory\n");
    return 0;
  }

  hipLaunchKernelGGL((testHalfMath), dim3(N), dim3(1), 0, 0, deviceArray);

  if (hipCallSuccessful(hipMemcpy(hostArray,
                                     deviceArray,
                                     N * sizeof(__half),
                                     hipMemcpyDeviceToHost)))
  {
    printf("Array content after kernel:\n");
    printArray(hostArray);
    printf("\n");
  }
  else
  {
    printf("Unable to copy memory from device to host\n");
  }

  hipFree(deviceArray);
  return 0;
}

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
//  1.8 Type Casting Intrinsics

#include <stdio.h>
#include <hip/hip_host_runtime_api.h>
#define N 10

__global__
void testTypeCasting(int *b)
{
  int i = blockIdx.x;
  long long ll = (long long)i;
  unsigned u = (unsigned) i;
  unsigned long long ull = (unsigned long long) i;
  float f = (float) i;
  double d = (double) i;
  if (i<N) {
    // 1.8 Type Casting Intrinsics
    b[i] = (int) __double2float_rd(d);
    b[i] += (int) __double2float_rn(d);
    b[i] += (int) __double2float_ru(d);
    b[i] += (int) __double2float_rz(d);
    b[i] += (int) __double2hiint(d);
    b[i] += (int) __double2int_rd(d);
    b[i] += (int) __double2int_rn(d);
    b[i] += (int) __double2int_ru(d);
    b[i] += (int) __double2int_rz(d);
    b[i] += (int) __double2ll_rd(d);
    b[i] += (int) __double2ll_rn(d);
    b[i] += (int) __double2ll_ru(d);
    b[i] += (int) __double2ll_rz(d);
    b[i] += (int) __double2loint(d);
    b[i] += (int) __double2uint_rd(d);
    b[i] += (int) __double2uint_rn(d);
    b[i] += (int) __double2uint_ru(d);
    b[i] += (int) __double2uint_rz(d);
    b[i] += (int) __double2ull_rd(d);
    b[i] += (int) __double2ull_rn(d);
    b[i] += (int) __double2ull_ru(d);
    b[i] += (int) __double2ull_rz(d);
    b[i] += (int) __double_as_longlong(d);
    b[i] += (int) __float2int_rd(f);
    b[i] += (int) __float2int_rn(f);
    b[i] += (int) __float2int_ru(f);
    b[i] += (int) __float2int_rz(f);
    b[i] += (int) __float2ll_rd(f);
    b[i] += (int) __float2ll_rn(f);
    b[i] += (int) __float2ll_ru(f);
    b[i] += (int) __float2ll_rz(f);
    b[i] += (int) __float2uint_rd(f);
    b[i] += (int) __float2uint_rn(f);
    b[i] += (int) __float2uint_ru(f);
    b[i] += (int) __float2uint_rz(f);
    b[i] += (int) __float2ull_rd(f);
    b[i] += (int) __float2ull_rn(f);
    b[i] += (int) __float2ull_ru(f);
    b[i] += (int) __float2ull_rz(f);
    b[i] += (int) __float_as_int(f);
    b[i] += (int) __float_as_uint(f);
    b[i] += (int) __hiloint2double(i,i);
    // b[i] += (int) __int2double_rn(i); // Fixme: missing function __nv_int2double_rn
    b[i] += (int) __int2float_rd(i);
    b[i] += (int) __int2float_rn(i);
    b[i] += (int) __int2float_ru(i);
    b[i] += (int) __int2float_rz(i);
    b[i] += (int) __int_as_float(i);
    b[i] += (int) __ll2double_rd(ll);
    b[i] += (int) __ll2double_rn(ll);
    b[i] += (int) __ll2double_ru(ll);
    b[i] += (int) __ll2double_rz(ll);
    b[i] += (int) __ll2float_rd(ll);
    b[i] += (int) __ll2float_rn(ll);
    b[i] += (int) __ll2float_ru(ll);
    b[i] += (int) __ll2float_rz(ll);
    b[i] += (int) __longlong_as_double(ll);
    b[i] += (int) __uint2double_rn(u);
    b[i] += (int) __uint2float_rd(u);
    b[i] += (int) __uint2float_rn(u);
    b[i] += (int) __uint2float_ru(u);
    b[i] += (int) __uint2float_rz(u);
    b[i] += (int) __uint_as_float(u);
    b[i] += (int) __ull2double_rd(ull);
    b[i] += (int) __ull2double_rn(ull);
    b[i] += (int) __ull2double_ru(ull);
    b[i] += (int) __ull2double_rz(ull);
    b[i] += (int) __ull2float_rd(ull);
    b[i] += (int) __ull2float_rn(ull);
    b[i] += (int) __ull2float_ru(ull);
    b[i] += (int) __ull2float_rz(ull);
  }
}

void printArray(int *array)
{
  printf("[");
  bool first = true;
  for (int i = 0; i<N; ++i)
  {
    if (first)
    {
      printf("%d", array[i]);
      first = false;
    }
    else
    {
      printf(", %d", array[i]);
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

  int hostArray[N];

  if (!haveComputeDevice())
  {
    printf("No compute device available\n");
    return 0;
  }

  for (int i = 0; i<N; ++i)
    hostArray[i] = 0.0;

  printf("Array content before kernel:\n");
  printArray(hostArray);
  printf("\n");

  int *deviceArray;
  if (!hipCallSuccessful(hipMalloc((void **)&deviceArray, N*sizeof(int))))
  {
    printf("Unable to allocate device memory\n");
    return 0;
  }

  hipLaunchKernelGGL((testTypeCasting), dim3(N), dim3(1), 0, 0, deviceArray);

  if (hipCallSuccessful(hipMemcpy(hostArray,
                                     deviceArray,
                                     N * sizeof(int),
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

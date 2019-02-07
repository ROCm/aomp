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
//  1.7 Integer Intrinsics

#include <stdio.h>
#include <hip/hip_host_runtime_api.h>
#define N 10

__global__
void testIntMath(int *b)
{
  int i = blockIdx.x;
  long long ll = (long long)i;
  unsigned u = (unsigned) i;
  unsigned long long ull = (unsigned long long) i;
  int f = (int) i;
  int dummy;
  int dummy2;
  int idummy;
  if (i<N) {
    // 1.7 Single Presicion Mathematical Functions

    b[i] = (int) __brev(u);
    b[i] += (int) __brevll(ull);
    // b[i] += (int) __byte_perm(u, u, u); // Fixme: missing function __nv_byte_perm
    b[i] += __clz(i);
    b[i] += __clzll(ll);
    b[i] += __ffs(i);
    b[i] += __ffsll(ll);
    //   b[i] += (int) __funnelshift_l(u, u, 2);  // Fixme: Add __funnelshift_l to cuda_open headers
    // b[i] += (int) __funnelshift_lc(u, u, 2);  // Fixme: Add __funnelshift_lc to cuda_open headers
    // b[i] += (int) __funnelshift_r(u, u, 2);  // Fixme: Add __funnelshift_r to cuda_open headers
    // b[i] += (int) __funnelshift_rc(u, u, 2);  // Fixme: Add __funnelshift_rc to cuda_open headers
    // b[i] += __hadd(i,i); // Fixme: missing function __nv_hadd
    b[i] += __mul24(i,i);
    b[i] += (int) __mul64hi(ll, ll);
    b[i] += __mulhi(i,i);
    b[i] += __popc(u);
    b[i] += __popcll(ull);
    // b[i] += __rhadd(i,i); // Fixme: missing function __nv_rhadd
    b[i] += (int) __sad(i, i, u);
    // b[i] += (int) __uhadd(u,u); // Fixme: missing function __uhadd
    b[i] += (int) __umul24(u, u);
    b[i] += (int) __umul64hi(u, u);
    b[i] += (int) __umulhi(u, u);
    // b[i] += (int) __urhadd(u, u); // Fixme: missing function __urhadd
    b[i] += (int) __usad(u, u, u);
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

  hipLaunchKernelGGL((testIntMath), dim3(N), dim3(1), 0, 0, deviceArray);

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

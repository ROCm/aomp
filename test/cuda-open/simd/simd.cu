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
//  1.9 SIMD Intrinsics

#include <stdio.h>
#include <hip/hip_host_runtime_api.h>
#define N 10

__global__
void testTypeCasting(unsigned *b)
{
  int i = blockIdx.x;
  unsigned u = (unsigned) i;
  if (i<N) {
    // 1.9 SIMD Intrinsics
    // b[i] = __vabs2(u); // Fixme: missing function __nv_vabs2
    // b[i] += __vabs4(u);  // Fixme: missing function __nv_vabs4
    // b[i] += __vabsdiffs2(u,u); // Fixme: missing function __nv_vabsdiffs2
    // b[i] += __vabsdiffs4(u,u); // Fixme: missing function __nv_vabsdiffs4
    // b[i] += __vabsdiffu2(u,u); // Fixme: missing function __nv_vabsdiffu2
    // b[i] += __vabsdiffu4(u,u); // Fixme: missing function __nv_vabsdiffu4
    // b[i] += __vabsss2(u); // Fixme: missing function __nv_vabsss2
    // b[i] += __vabsss4(u); // Fixme: missing function __nv_vabsss4
    // b[i] += __vadd2(u,u); // Fixme: missing function __nv_vadd2
    // b[i] += __vadd4(u,u); // Fixme: missing function __nv_vadd4
    // b[i] += __vaddss2(u,u); // Fixme: missing function __nv_vaddss2
    // b[i] += __vaddss4(u,u); // Fixme: missing function __nv_vaddss4
    // b[i] += __vaddus2(u,u); // Fixme: missing function __nv_vaddus2
    // b[i] += __vaddus4(u,u); // Fixme: missing function __nv_vaddus4
    // b[i] += __vavgs2(u,u); // Fixme: missing function __nv_vavgs2
    // b[i] += __vavgs4(u,u); // Fixme: missing function __nv_vavgs4
    // b[i] += __vavgu2(u,u); // Fixme: missing function __nv_vavgu2
    // b[i] += __vavgu4(u,u); // Fixme: missing function __nv_vavgu4
    // b[i] += __vcmpeq2(u,u); // Fixme: missing function __nv_vcmpeq2
    // b[i] += __vcmpeq4(u,u); // Fixme: missing function __nv_vcmpeq4
    // b[i] += __vcmpges2(u,u); // Fixme: missing function __nv_vcmpges2
    // b[i] += __vcmpges4(u,u); // Fixme: missing function __nv_vcmpges4
    // b[i] += __vcmpgeu2(u,u); // Fixme: missing function __nv_vcmpgeu2
    // b[i] += __vcmpgeu4(u,u); // Fixme: missing function __nv_vcmpgeu4
    // b[i] += __vcmpgts2(u,u); // Fixme: missing function __nv_vcmpgts2
    // b[i] += __vcmpgts4(u,u); // Fixme: missing function __nv_vcmpgts4
    // b[i] += __vcmpgtu2(u,u); // Fixme: missing function __nv_vcmpgtu2
    // b[i] += __vcmpgtu4(u,u); // Fixme: missing function __nv_vcmpgtu4
    // b[i] += __vcmples2(u,u); // Fixme: missing function __nv_vcmples2
    // b[i] += __vcmples4(u,u); // Fixme: missing function __nv_vcmples4
    // b[i] += __vcmpleu2(u,u); // Fixme: missing function __nv_vcmpleu2
    // b[i] += __vcmpleu4(u,u); // Fixme: missing function __nv_vcmpleu4
    // b[i] += __vcmplts2(u,u); // Fixme: missing function __nv_vcmplts2
    // b[i] += __vcmplts4(u,u); // Fixme: missing function __nv_vcmplts4
    // b[i] += __vcmpltu2(u,u); // Fixme: missing function __nv_vcmpltu2
    // b[i] += __vcmpltu4(u,u); // Fixme: missing function __nv_vcmpltu4
    // b[i] += __vcmpne2(u,u); // Fixme: missing function __nv_vcmpne2
    // b[i] += __vcmpne4(u,u); // Fixme: missing function __nv_vcmpne4
    // b[i] += __vhaddu2(u,u); // Fixme: missing function __nv_vhaddu2
    // b[i] += __vhaddu4(u,u); // Fixme: missing function __nv_vhaddu4
    // b[i] += __vmaxs2(u,u); // Fixme: missing function __nv_vmaxs2
    // b[i] += __vmaxs4(u,u); // Fixme: missing function __nv_vmaxs4
    // b[i] += __vmaxu2(u,u); // Fixme: missing function __nv_vmaxu2
    // b[i] += __vmaxu4(u,u); // Fixme: missing function __nv_vmaxu4
    // b[i] += __vmins2(u,u); // Fixme: missing function __nv_vmins2
    // b[i] += __vmins4(u,u); // Fixme: missing function __nv_vmins4
    // b[i] += __vminu2(u,u); // Fixme: missing function __nv_vminu2
    // b[i] += __vminu4(u,u); // Fixme: missing function __nv_vminu4
    // b[i] += __vneg2(u); // Fixme: missing function __nv_vneg2
    // b[i] += __vneg4(u); // Fixme: missing function __nv_vneg4
    // b[i] += __vnegss2(u); // Fixme: missing function __nv_vnegss2
    // b[i] += __vnegss4(u); // Fixme: missing function __nv_vnegss4
    // b[i] += __vsads2(u,u); // Fixme: missing function __nv_vsads2
    // b[i] += __vsads4(u,u); // Fixme: missing function __nv_vsads4
    // b[i] += __vsadu2(u,u); // Fixme: missing function __nv_vsadu2
    // b[i] += __vsadu4(u,u); // Fixme: missing function __nv_vsadu4
    // b[i] += __vseteq2(u,u); // Fixme: missing function __nv_vseteq2
    // b[i] += __vseteq4(u,u); // Fixme: missing function __nv_vseteq4
    // b[i] += __vseteq2(u,u); // Fixme: missing function __nv_vseteq2
    // b[i] += __vseteq4(u,u); // Fixme: missing function __nv_vseteq4
    // b[i] += __vsetges2(u,u); // Fixme: missing function __nv_vsetges2
    // b[i] += __vsetges4(u,u); // Fixme: missing function __nv_vsetges4
    // b[i] += __vsetgeu2(u,u); // Fixme: missing function __nv_vsetgeu2
    // b[i] += __vsetgeu4(u,u); // Fixme: missing function __nv_vsetgeu4
    // b[i] += __vsetgts2(u,u); // Fixme: missing function __nv_vsetgts2
    // b[i] += __vsetgts4(u,u); // Fixme: missing function __nv_vsetgts4
    // b[i] += __vsetgtu2(u,u); // Fixme: missing function __nv_vsetgtu2
    // b[i] += __vsetgtu4(u,u); // Fixme: missing function __nv_vsetgtu4
    // b[i] += __vsetles2(u,u); // Fixme: missing function __nv_vsetles2
    // b[i] += __vsetles4(u,u); // Fixme: missing function __nv_vsetles4
    // b[i] += __vsetleu2(u,u); // Fixme: missing function __nv_vsetleu2
    // b[i] += __vsetleu4(u,u); // Fixme: missing function __nv_vsetleu4
    // b[i] += __vsetlts2(u,u); // Fixme: missing function __nv_vsetlts2
    // b[i] += __vsetlts4(u,u); // Fixme: missing function __nv_vsetlts4
    // b[i] += __vsetltu2(u,u); // Fixme: missing function __nv_vsetltu2
    // b[i] += __vsetltu4(u,u); // Fixme: missing function __nv_vsetltu4
    // b[i] += __vsetne2(u,u); // Fixme: missing function __nv_vsetne2
    // b[i] += __vsetne4(u,u); // Fixme: missing function __nv_vsetne4
    // b[i] += __vsub2(u,u); // Fixme: missing function __nv_vsub2
    // b[i] += __vsub4(u,u); // Fixme: missing function __nv_vsub4
    // b[i] += __vsubss2(u,u); // Fixme: missing function __nv_vsubss2
    // b[i] += __vsubss4(u,u); // Fixme: missing function __nv_vsubss4
    // b[i] += __vsubus2(u,u); // Fixme: missing function __nv_vsubus2
    // b[i] += __vsubus4(u,u); // Fixme: missing function __nv_vsubus4



  }
}

void printArray(unsigned *array)
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

  unsigned hostArray[N];

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

  unsigned *deviceArray;
  if (!hipCallSuccessful(hipMalloc((void **)&deviceArray, N*sizeof(int))))
  {
    printf("Unable to allocate device memory\n");
    return 0;
  }

  hipLaunchKernelGGL((testTypeCasting), dim3(N), dim3(1), 0, 0, deviceArray);

  if (hipCallSuccessful(hipMemcpy(hostArray,
                                     deviceArray,
                                     N * sizeof(unsigned),
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

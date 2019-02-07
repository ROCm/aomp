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
//  1.3 Single Presicion Mathematical Functions
//  1.5 Single Presicion Intrinsics
#include <stdio.h>
#include <hip/hip_host_runtime_api.h>
#define N 10

__global__
void testFloatMath(float *b)
{
  int i = blockIdx.x;
  float f = (float) i;
  float dummy;
  float dummy2;
  int idummy;
  if (i<N) {
    // 1.3 Single Presicion Mathematical Functions
    b[i] = acosf(f);
    b[i] += acoshf(f);
    b[i] += asinf(f);
    b[i] += asinhf(f);
    b[i] += atan2f(f,f);
    b[i] += atanf(f);
    b[i] += atanhf(f);
    b[i] += cbrtf(f);
    b[i] += ceilf(f);
    b[i] += copysignf(f, -f);
    b[i] += cosf(f);
    b[i] += coshf(f);
    b[i] += cospif(f);
    b[i] += cyl_bessel_i0f(f);
    b[i] += cyl_bessel_i1f(f);
    b[i] += erfcf(f);
    b[i] += erfcinvf(f);
    b[i] += erfcxf(f);
    b[i] += erff(f);
    b[i] += erfinvf(f);
    b[i] += exp10f(f);
    b[i] += exp2f(f);
    b[i] += expf(f);
    b[i] += expm1f(f);
    b[i] += fabsf(f);
    b[i] += fdimf(f,f);
    b[i] += fdividef(f,f);
    b[i] += floorf(f);
    b[i] += fmaf(f,f,f);
    b[i] += fmaxf(f,f);
    b[i] += fminf(f,f);
    b[i] += fmodf(f,f);
    b[i] += frexpf(f, &idummy);
    b[i] += hypotf(f,f);
    b[i] += (float) ilogbf(f);
    b[i] += isfinite(f);
    b[i] += isinf(f);
    b[i] += isnan(f);
    b[i] += j0f(f);
    b[i] += j1f(f);
    // b[i] += jnf(1,f); // Fixme: missing function _nv_jnf, no corresponding function in ocml.
    b[i] += ldexpf(f,1);
    b[i] += lgammaf(f);
    b[i] += (float) llrintf(f);
    b[i] += (float) llroundf(f);
    b[i] += log10f(f);
    b[i] += log1pf(f);
    b[i] += log2f(f);
    b[i] += logbf(f);
    b[i] += logf(f);
//    b[i] += (float) lrintf(f);
    b[i] += (float) lroundf(f);
    b[i] += modff(f, &dummy); 
    // b[i] += nanf(""); // Fixme: Add to cuda_open headers, need to convert unsigned value string to unsigned int and call __ocml_nan_f32.
    b[i] += nearbyintf(f);
    b[i] += nextafterf(f,f);
    b[i] += norm3df(f,f,f);
    b[i] += norm4df(f,f,f,f);
    b[i] += normcdff(f);
    b[i] += normcdfinvf(f);
    // b[i] += normf(1,&f); // Fixme: missing function __nv_normf, no corresponding function in ocml.
    b[i] += powf(f,f);
    b[i] += rcbrtf(f);
    b[i] += remainderf(f,f);
    b[i] += remquof(f,f, &idummy);
    b[i] += rhypotf(f,f);
    b[i] += rintf(f);
    b[i] += rnorm3df(f,f,f);
    b[i] += rnorm4df(f,f,f,f);
    // b[i] += rnormf(1, &f); // Fixme: missing function __nv_rnormf, no corresponding function in ocml
    b[i] += roundf(f);
    b[i] += rsqrtf(f);
    b[i] += scalblnf(f, 1);
    b[i] += scalbnf(f, 1);
    b[i] += signbit(f);
    sincosf(f, &dummy, &dummy2);
    sincospif(f, &dummy, &dummy2);
    b[i] += sinf(f);
    b[i] += sinhf(f);
    b[i] += sinpif(f);
    b[i] += sqrtf(f);
    b[i] += tanf(f);
    b[i] += tanhf(f);
    b[i] += tgammaf(f);
    b[i] += truncf(f);
    b[i] += y0f(f);
    b[i] += y1f(f);
    // b[i] += ynf(1,f); // Fixme: missing function __nv_ynf, no corresponding function in ocml

   // 1.5 Single Presicion Intrinsics

    b[i] += __cosf(f);
    b[i] += __exp10f(f);
    b[i] += __expf(f);
    //   b[i] += __fadd_rd(f, f); // LLVM does not support non-standard rounding modes.
    //    b[i] += __fadd_rn(f, f); // LLVM unsupported rounding mode
    //    b[i] += __fadd_ru(f, f); // LLVM unsupported rounding mode
    //    b[i] += __fadd_rz(f, f); // LLVM unsupported rounding mode
    //    b[i] += __fdiv_rd(f, f); // LLVM unsupported rounding mode
    //    b[i] += __fdiv_rn(f, f); // LLVM unsupported rounding mode
    //    b[i] += __fdiv_ru(f, f); // LLVM unsupported rounding mode
    //    b[i] += __fdiv_rz(f, f); // LLVM unsupported rounding mode
    //    b[i] += __fdividef(f, f); // Undefined symbol native_div
    // b[i] += __fmaf_rd(f, f, f); // LLVM unsupported rounding mode
    // b[i] += __fmaf_rn(f, f, f); // LLVM unsupported rounding mode
    // b[i] += __fmaf_ru(f, f, f); // LLVM unsupported rounding mode
    // b[i] += __fmaf_rz(f, f, f); // LLVM unsupported rounding mode
    // b[i] += __fmul_rd(f, f); // LLVM unsupported rounding mode
    // b[i] += __fmul_rn(f, f); // LLVM unsupported rounding mode
    // b[i] += __fmul_ru(f, f); // LLVM unsupported rounding mode
    // b[i] += __fmul_rz(f, f); // LLVM unsupported rounding mode
    // b[i] += __frcp_rd(f); // LLVM unsupported rounding mode
    // b[i] += __frcp_rn(f); // LLVM unsupported rounding mode
    // b[i] += __frcp_ru(f); // LLVM unsupported rounding mode
    // b[i] += __frcp_rz(f); // LLVM unsupported rounding mode
    // b[i] += __fsqrt_rd(f); // LLVM unsupported rounding mode
    // b[i] += __fsqrt_rn(f); // LLVM unsupported rounding mode
    // b[i] += __fsqrt_ru(f); // LLVM unsupported rounding mode
    // b[i] += __fsqrt_rz(f); // LLVM unsupported rounding mode
    // b[i] += __fsub_rd(f, f); // LLVM unsupported rounding mode
    b[i] += __log10f(f);
    b[i] += __log2f(f);
    b[i] += __logf(f);
    b[i] += __powf(f, f);
    b[i] += __saturatef(f);
    __sincosf(f, &dummy, &dummy2);
    b[i] += __sinf(f);
    b[i] += __tanf(f);
  }
}

void printArray(float *array)
{
  printf("[");
  bool first = true;
  for (int i = 0; i<N; ++i)
  {
    if (first)
    {
      printf("%f", array[i]);
      first = false;
    }
    else
    {
      printf(", %f", array[i]);
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

  float hostArray[N];

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

  float *deviceArray;
  if (!hipCallSuccessful(hipMalloc((void **)&deviceArray, N*sizeof(float))))
  {
    printf("Unable to allocate device memory\n");
    return 0;
  }

  hipLaunchKernelGGL((testFloatMath), dim3(N), dim3(1), 0, 0, deviceArray);

  if (hipCallSuccessful(hipMemcpy(hostArray,
                                     deviceArray,
                                     N * sizeof(float),
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

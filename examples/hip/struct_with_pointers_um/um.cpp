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

// This program replicates OpenMP behavior for auto zero-copy found in
// aomp/test/smoke-fails/struct_with_ptrs_zero_copy


#include <cstdlib>
#include <cstdio>
#include <sys/time.h>

#include "hip/hip_runtime.h"

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

typedef struct {
  double x, y, z;
} Monomer;

typedef struct {
  struct {
    size_t size;
    Monomer *buf, *dev_buf;
  } all_Monomer;
} Allocator;

typedef struct {
  Allocator *alloc;
} Phase;

int main() {
  size_t n = 1600000000;
  Allocator alloc;
  Phase phase;
  Phase *const p = &phase;
  p->alloc = &alloc;

  p->alloc->all_Monomer.size = n;
  p->alloc->all_Monomer.buf = (Monomer *)malloc(n*sizeof(Monomer));

  //p->alloc->all_Monomer.dev_buf = (Monomer *)omp_target_alloc(n*sizeof(Monomer), 0);
  hipMalloc(&(p->alloc->all_Monomer.dev_buf), n*sizeof(Monomer));

  // not needed, performed by hipMemcpy below
  //  omp_target_associate_ptr(p->alloc->all_Monomer.buf, p->alloc->all_Monomer.dev_buf, n*sizeof(Monomer), 0, 0);

  // not needed, just copying the memory for buf to device_buf
  //#pragma omp target enter data map(to: p[:1])
  //#pragma omp target enter data map(to: p->alloc[:1])

  struct timeval t1, t2;
  double td;
  size_t bytes =
      sizeof(p->alloc->all_Monomer.buf[0]) * p->alloc->all_Monomer.size;
  printf("Preparing to update to() at file: %s line: %d: %lu bytes\n", __FILE__,
         __LINE__, bytes);
  gettimeofday(&t1, 0);

  //#pragma omp target update to(p->alloc->all_Monomer.buf[:p->alloc->all_Monomer.size])
  hipMemcpy(p->alloc->all_Monomer.dev_buf, p->alloc->all_Monomer.buf, p->alloc->all_Monomer.size*sizeof(Monomer), hipMemcpyHostToDevice);

  gettimeofday(&t2, 0);
  td = (t2.tv_sec + t2.tv_usec / 1e6) - (t1.tv_sec + t1.tv_usec / 1e6);
  printf("Update to(): %lu bytes %g seconds %g MB/sec\n", bytes, td,
         1e-6 * bytes / td);

  return 0;
}

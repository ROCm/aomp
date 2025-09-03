// Copyright © Advanced Micro Devices, Inc., or its affiliates.
//
// SPDX-License-Identifier:  MIT

#include <omp.h>

#define LDS_ATTR volatile __attribute__((address_space(3)))
#define SHMEM_SIZE 256

int main(int argc, char *argv[]) {
  int N = SHMEM_SIZE;
  int *Ptr = new int[N];
#pragma omp target data map(tofrom : Ptr[0 : N])
#pragma omp target
  {
    LDS_ATTR static int DPtr[SHMEM_SIZE];
    // #pragma omp parallel for
    for (int i = 0; i <= N; ++i) {
      DPtr[i] = 2 * (i + 1);
      Ptr[i] = DPtr[i];
    }
  }
  delete[] Ptr;
  return 0;
}

/// CHECK:=================================================================
/// CHECK-NEXT:=={{[0-9]+}}==ERROR: AddressSanitizer: heap-buffer-overflow on amdgpu device 0 at pc [[PC:0x[0-9a-f]+]]
/// CHECK-NEXT:WRITE of size 4 in workgroup id ({{[0-9]+}},0,0)
/// CHECK-NEXT:  #0 [[PC]] in __omp_offloading_{{.*}} at {{.*}}aomp/test/smoke-asan/omp-static-direct-lbo/omp-static-direct-lbo.cpp:15:{{[0-9]+}}
/// CHECK:{{0x[0-9a-f]+}} is 1056 bytes above an address from a device malloc (or free) call of size 1312 from
/// CHECK-NEXT:  #0 0xfffffffffffffffc in ?? at ??:0:0

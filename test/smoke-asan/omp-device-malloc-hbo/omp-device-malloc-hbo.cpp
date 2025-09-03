// Copyright © Advanced Micro Devices, Inc., or its affiliates.
//
// SPDX-License-Identifier:  MIT

#include <omp.h>

int main(int argc, char *argv[]) {
  int N = 1000;
  int *Ptr = new int[N];
#pragma omp target data map(from : Ptr[0 : N])
#pragma omp target teams num_teams(1)
  {
    int *DPtr = (int *)malloc(N * sizeof(int));
#pragma omp distribute parallel for
    for (int i = 0; i <= N; i++) {
      DPtr[i] = 2 * (i + 1);
      Ptr[i] = DPtr[i];
    }
  }
  delete[] Ptr;
  return 0;
}

/// CHECK:=================================================================
/// CHECK-NEXT:=={{[0-9]+}}==ERROR: AddressSanitizer: heap-buffer-overflow on amdgpu device 0 at pc [[PC:.*]]
/// CHECK-NEXT:WRITE of size 4 in workgroup id ({{[0-9]+}},0,0)
/// CHECK-NEXT:  #0 [[PC]] in __omp_offloading_{{.*}} at {{.*}}aomp/test/smoke-asan/omp-device-malloc-hbo/omp-device-malloc-hbo.cpp:12:{{[0-9]+}}
/// CHECK-NEXT: (inlined by) __omp_offloading_{{.*}} at {{.*}}aomp/test/smoke-asan/omp-device-malloc-hbo/omp-device-malloc-hbo.cpp:10:{{[0-9]+}}
/// CHECK:{{0[xX][0-9a-fA-F]+}} is 4000 bytes above an address from a device malloc (or free) call of size 4000 from
/// CHECK-NEXT:  #0 {{0[xX][0-9a-fA-F]+}} in __omp_offloading_{{.*}} at {{.*}}aomp/test/smoke-asan/omp-device-malloc-hbo/omp-device-malloc-hbo.cpp:9:{{[0-9]+}}
/// CHECK-NEXT: (inlined by) __omp_offloading_{{.*}} at {{.*}}aomp/test/smoke-asan/omp-device-malloc-hbo/omp-device-malloc-hbo.cpp:7:{{[0-9]+}}

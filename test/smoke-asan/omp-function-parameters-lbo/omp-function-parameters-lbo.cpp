// Copyright © Advanced Micro Devices, Inc., or its affiliates.
//
// SPDX-License-Identifier:  MIT

#include <omp.h>

#define LDS_ATTR volatile __attribute__((address_space(3)))
#define SHMEM_SIZE 256

void OffloadToGpu(LDS_ATTR float *DPtr, float *Ptr, int N) {
#pragma omp parallel for
  for (int i = 0; i <= N; i++) {
    DPtr[i] = 2 * (i + 1);
    Ptr[i] = DPtr[i];
  }
}

int main(int argc, char *argv[]) {
  int N = SHMEM_SIZE;
  float *Ptr = new float[N];
#pragma omp target data map(tofrom : Ptr[0 : N])
#pragma omp target ompx_dyn_cgroup_mem(SHMEM_SIZE * sizeof(float))
  {
    OffloadToGpu((LDS_ATTR float *)llvm_omp_target_dynamic_shared_alloc(), Ptr,
		 N);
  }
  delete[] Ptr;
  return 0;
}

/// CHECK:=================================================================
/// CHECK-NEXT:=={{[0-9]+}}==ERROR: AddressSanitizer: heap-buffer-overflow on amdgpu device 0 at pc [[PC:.*]]
/// CHECK-NEXT:WRITE of size 4 in workgroup id ({{[0-9]+}},0,0)
/// CHECK-NEXT:  #0 [[PC]] in {{.*}}OffloadToGpu{{.*}} at {{.*}}aomp/test/smoke-asan/omp-function-parameters-lbo/omp-function-parameters-lbo.cpp:9:{{[0-9]+}}
/// CHECK-NEXT: (inlined by) OffloadToGpu(float volatile AS3*, float*, int) at {{.*}}aomp/test/smoke-asan/omp-function-parameters-lbo/omp-function-parameters-lbo.cpp:7:{{[0-9]+}}
/// CHECK-NEXT: (inlined by) __omp_offloading_{{.*}} at {{.*}}aomp/test/smoke-asan/omp-function-parameters-lbo/omp-function-parameters-lbo.cpp:20:{{[0-9]+}}
/// CHECK:{{0x[0-9a-f]+}} is 1184 bytes above an address from a device malloc (or free) call of size 1184 from
/// CHECK-NEXT:  #0 0xfffffffffffffffc in ?? at ??:0:0

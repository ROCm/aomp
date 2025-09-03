// Copyright © Advanced Micro Devices, Inc., or its affiliates.
//
// SPDX-License-Identifier:  MIT

#include <cstdio>

// When OMPX_DGPU_MAPS=0 and HSA_XNACK=1, it performs zero-copy on APU.

int main() {
  int n = 1024;
  int *a = new int[n]; // 4096 bytes
  int k = 3;
  int b[n]; // 4096 bytes

  for (int i = 0; i < n; i++)
    b[i] = i;

/// CHECK-NOT: data_submit_async: {{.*}} 0 ({{.*}} 4096, {{.*}})
/// CHECK-NOT: data_submit_async: {{.*}} 0 ({{.*}} 4096, {{.*}})  
#pragma omp target teams distribute parallel for map(tofrom : a[ : n]) map(to : b[ : n])
  for (int i = 0; i < n; i++)
    a[i] = i + b[i] + k;

/// CHECK-NOT: data_retrieve_async: {{.*}} 0 ({{.*}} 4096, {{.*}})

  int err = 0;
  for (int i = 0; i < n; i++)
    if (a[i] != i + b[i] + k)
      err++;

  /// CHECK: PASS
  if (err == 0)
    printf("PASS\n");
  return err;
}

// Copyright © Advanced Micro Devices, Inc., or its affiliates.
//
// SPDX-License-Identifier:  MIT

#include <omp.h>
#include <stdio.h>

int main() {
  int N = 10000;

  _Float16 a[N];
  __bf16 b[N];
  short c[N];

  for (int i = 0; i < N; i++) {
    a[i] = i;
    b[i] = i;
    c[i] = i;
  }

  _Float16 min1 = 10;
  __bf16 min2 = 11;
  short min3 = 12;

  _Float16 max1 = 0;
  __bf16 max2 = 0;
  short max3 = -10;

#pragma omp target teams distribute parallel for reduction(min:min1)
  for (int j = 0; j < N; j = j + 1)
    min1 = __builtin_fmin(min1, a[j]);

#pragma omp target teams distribute parallel for reduction(min:min2)
  for (int j = 0; j < N; j = j + 2)
    min2 = __builtin_fmin(min2, b[j]);

#pragma omp target teams distribute parallel for reduction(min:min3)
  for (int j = 0; j < N; j = j + 3)
    min3 = __builtin_fmin(c[j], min3);

#pragma omp target teams distribute parallel for reduction(max : max1)
  for (int j = 0; j < N; j = j + 1)
    max1 = __builtin_fmax(max1, a[j]);

#pragma omp target teams distribute parallel for reduction(max : max2)
  for (int j = 0; j < N; j = j + 2)
    max2 = __builtin_fmax(max2, b[j]);

#pragma omp target teams distribute parallel for reduction(max : max3)
  for (int j = 0; j < N; j = j + 3)
    max3 = __builtin_fmax(c[j], max3);

  printf("min1=%f min2=%f min3=%hd max1=%f max2=%f max3=%hd\n", (float)min1, (float)min2, min3, (float)max1, (float)max2, max3);

  int rc = (min1 != 0) || (min2 != 0) || (min3 != 0) || (max1 != 10000) || (max2 != 9984) || (max3 != 9999);

  if (!rc)
    printf("Success\n");

  return rc;
}

/// CHECK: DEVID:[[S:[ ]*]][[DEVID:[0-9]+]] SGN:8
/// CHECK: DEVID:[[S:[ ]*]][[DEVID:[0-9]+]] SGN:8
/// CHECK: DEVID:[[S:[ ]*]][[DEVID:[0-9]+]] SGN:8
/// CHECK: DEVID:[[S:[ ]*]][[DEVID:[0-9]+]] SGN:8
/// CHECK: DEVID:[[S:[ ]*]][[DEVID:[0-9]+]] SGN:8
/// CHECK: DEVID:[[S:[ ]*]][[DEVID:[0-9]+]] SGN:8

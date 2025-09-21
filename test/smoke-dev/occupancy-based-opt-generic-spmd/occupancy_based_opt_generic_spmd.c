// Copyright © Advanced Micro Devices, Inc., or its affiliates.
//
// SPDX-License-Identifier:  MIT

#define N 1048576
#define NUM_THREADS 256
#define NUM_DISTR (N / NUM_THREADS)

#include <stdio.h>

int main() {
  int a[N];
  int i, j;

#pragma omp target teams distribute
  for (i = 0; i < NUM_DISTR; i++)
#pragma omp parallel for
    for (j = 0; j < NUM_THREADS; j++)
      a[i * NUM_THREADS + j] = (i * NUM_THREADS + j);

  for (i = 0; i < N; i++)
    if (a[i] != i) {
      printf("wrong value: a[%d]=%d\n", i, a[i]);
      return 1;
    }
  printf("Success\n");
  return 0;
}

/// CHECK: Achieved Occupancy: 100%

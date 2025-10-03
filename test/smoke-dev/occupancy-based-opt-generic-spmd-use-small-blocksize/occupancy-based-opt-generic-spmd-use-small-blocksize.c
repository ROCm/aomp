// Copyright © Advanced Micro Devices, Inc., or its affiliates.
//
// SPDX-License-Identifier:  MIT

#define N 4194304
#define NUM_THREADS 256
#define NUM_DISTR (N / NUM_THREADS)

#include <stdio.h>
#include <stdlib.h>

int main() {
  int *a = (int*)malloc(sizeof(int)*N);
  int i, j;

#pragma omp target teams distribute map(tofrom:a[0:N])
  for (i = 0; i < NUM_DISTR; i++)
#pragma omp parallel for
    for (j = 0; j < NUM_THREADS; j++)
      a[i * NUM_THREADS + j] = (i * NUM_THREADS + j);

  for (i = 0; i < N; i++)
    if (a[i] != i) {
      printf("wrong value: a[%d]=%d\n", i, a[i]);
      free(a);
      return 1;
    }
  printf("Success\n");
  free(a);
  return 0;
}

/// CHECK: SGN:3
/// CHECK: teamsXthrds:([[S:[ ]*]][[NUM_TEAMS:[0-9]+]]X  64)
/// CHECK: Achieved Occupancy: 100%


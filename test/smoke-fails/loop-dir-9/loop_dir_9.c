#include <omp.h>
#include <stdio.h>
#include <stdlib.h>

#define N 64

int main() {
  int errors = 0;
  int x = 0;
  int device_result[N] = {0};
  int result[N] = {0};

  for (int i = 0; i < N; i++) {
    result[i] = 2 * i ;
  }

#pragma omp target parallel loop num_threads(N) uses_allocators(omp_pteam_mem_alloc) allocate(omp_pteam_mem_alloc: x) private(x) map(from: device_result)
  for (int i = 0; i < N; i++) {
    x = omp_get_thread_num();
    device_result[i] = i + x;
  }

  for (int i = 0; i < N; i++) {
    if (result[i] != device_result[i])
      errors++;
  }
  if (!errors)
    printf("Success\n");
  return errors;
}

/// CHECK: DEVID:[[S:[ ]*]][[DEVID:[0-9]+]] SGN:2

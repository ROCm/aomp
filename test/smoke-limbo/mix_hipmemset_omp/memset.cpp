#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include "hip_memset.h"

#define N 1000

int main() {
  int n = N;
  int dev = omp_get_default_device();
  int host = omp_get_initial_device();
  int *a = (int *) omp_target_alloc(n * sizeof(int), dev);
  set_mem(a, n);

  // a is device memory, so it cannot be read from host code. Whether such a
  // read happens to work is decided by how much of the GPU memory the CPU can
  // map, which varies from machine to machine.
  int *b = (int *) malloc(n * sizeof(int));
  if (omp_target_memcpy(b, a, n * sizeof(int), 0, 0, host, dev)) {
    printf("omp_target_memcpy failed\n");
    return 1;
  }

  int err = 0;
  for(int i = 0; i < n; i++)
    if (b[i] != 0) {
      printf("Error at %d: a[%d] = %d\n", i, i, b[i]);
      err++;
      if (err > 10) break;
    }

  free(b);
  omp_target_free(a, dev);
  return err;
}

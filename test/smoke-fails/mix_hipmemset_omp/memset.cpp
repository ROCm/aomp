#include <stdio.h>
#include <omp.h>
#include "hip_memset.h"

#define N 1000

int main() {
  int n = N;
  int *a = (int *) omp_target_alloc(n * sizeof(int), omp_get_default_device());
  set_mem(a, n);
  int err = 0;
  for(int i = 0; i < n; i++)
    if (a[i] != 0) {
      printf("Error at %d: a[%d] = %d\n", i, i, a[i]);
      err++;
      if (err > 10) break;
    }
  return err;
}

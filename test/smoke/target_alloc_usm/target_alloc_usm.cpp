#include <cstdio>
#include <omp.h>

#define N 123456

#pragma omp requires unified_shared_memory

int main() {
  int n = N;
  double * a = (double *)omp_target_alloc(n*sizeof(double), 0);
  double * b = new double[n];

  #pragma omp target teams distribute parallel for map(b[:n]) device(1)
  for(int i = 0; i < n; i++)
    a[i] = b[i];

  // check
  int err = 0;
  for(int i = 0; i < n; i++)
    if (a[i] != b[i]) {
      err++;
      printf("Error at %d: should be %lf, got %lf\n", i, (double)i, a[i]);
      if (err > 10) break;
    }

  return err;
}

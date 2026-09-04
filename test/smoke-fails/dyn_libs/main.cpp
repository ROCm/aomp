#include <cstdio>
#include "libA.h"
#include "libB.h"

#define N 100000

#pragma omp requires unified_shared_memory

int main() {
  size_t n = N;
  double *a = new double[n];
  double *b = new double[n];

  set_to_zero(a, n);
  
  #pragma omp target teams distribute parallel for map(from:b[0:n]) map(to:a[0:n])
  for(size_t i = 0; i < n; i++) {
    b[i] = a[i];
  }

  set_to_one(b, n);

  int err = 0;
  for(size_t i = 0; i < n; i++)
    if (b[i] != 1.0) {
      printf("%zu: got %lf, expected %lf\n", i, b[i], a[i]);
      err++;
      if (err > 10) return err;
    }

  return err;
}

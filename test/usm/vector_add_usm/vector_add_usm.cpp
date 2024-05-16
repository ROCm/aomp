#include <cstdio>

#pragma omp requires unified_shared_memory

int main()
{
  const size_t n = 1024*100;

  double *a = new double[n];
  double *b = new double[n];
  double *c = new double[n];

  // initialize
  for(size_t i = 0; i < n; i++) {
    a[i] = -1;
    b[i] = i;
    c[i] = 2*i;
  }

  #pragma omp target teams loop
  for(size_t i = 0; i < n; i++) {
    a[i] = b[i] + c[i];
  }

  int err = 0;
  for(size_t i = 0; i < n; i++)
    if (a[i] != b[i]+c[i]) {
      printf("Error at %zu: got %lf, expected %lf\n", i, a[i], b[i]+c[i]);
      if (err > 10) return err;
    }

  if (!err)
    printf("Success\n");

  return err;
}

#include <unistd.h>
#include <cstdio>
#include <omp.h>

#define N 123456

int main() {
  int err = 0;
  int n = N;

  // allocator for locked memory with predefined allocator
  double *a = (double *)omp_alloc(n*sizeof(double), ompx_pinned_mem_alloc);
  double *b = (double *)omp_alloc(n*sizeof(double), ompx_pinned_mem_alloc);
  //double *a = new double[n];
  //double *b = new double[n];

  for(int i = 0; i < n; i++) {
    a[i] = 0;
    b[i] = i;
  }

  #pragma omp target teams distribute parallel for map(to:b[:n]) map(from:a[:n])
  for(int i = 0; i < n; i++) {
    a[i] = b[i];
  }
  //  sleep(5);
  for(int i = 0; i < n; i++)
    if (a[i] != b[i]) {
      err++;
      printf("Error at %d, expected %lf, got %lf\n", i, b[i], a[i]);
      if (err > 10) return err;
    }

  omp_free(a, ompx_pinned_mem_alloc);
  omp_free(b, ompx_pinned_mem_alloc);

  return err;
}

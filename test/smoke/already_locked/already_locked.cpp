#include <unistd.h>
#include <cstdio>
#include <omp.h>

#define N 123456

int main() {
  int err = 0;
  int n = N;

  // allocator for locked memory
  omp_alloctrait_t pinned_trait[1] = {{omp_atk_pinned, omp_atv_true}};
  omp_allocator_handle_t pinned_alloc = omp_init_allocator(omp_default_mem_space, 1, pinned_trait);
  double *a = (double *)omp_alloc(n*sizeof(double), pinned_alloc);
  double *b = (double *)omp_alloc(n*sizeof(double), pinned_alloc);

  for(int i = 0; i < n; i++) {
    a[i] = 0;
    b[i] = i;
  }

  #pragma omp target teams distribute parallel for map(to:b[:n]) map(from:a[:n])
  for(int i = 0; i < n; i++) {
    a[i] = b[i];
  }
 
  #pragma omp target teams distribute parallel for map(to:b[:n]) map(tofrom:a[:n])
  for(int i = 0; i < n; i++) {
    a[i] += b[i];
  }

  for(int i = 0; i < n; i++)
    if (a[i] != 2.0*b[i]) {
      err++;
      printf("Error at %d, expected %lf, got %lf\n", i, 2.0*b[i], a[i]);
      if (err > 10) return err;
    }

  omp_free(a, pinned_alloc);
  omp_free(b, pinned_alloc);
  
  return err;
}

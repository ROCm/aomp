#include<cstdio>
#include<omp.h>

#define N 1000000
#define T 10000

int main() {
  const int n = N;
  const int times = T;
  omp_alloctrait_t pinned_trait[1] = {{omp_atk_pinned, omp_atv_true}};
  omp_allocator_handle_t pinned_alloc = omp_init_allocator(omp_default_mem_space, 1, pinned_trait);
  for(int t = 0; t < times; t++) {
    #pragma omp parallel num_threads(128)
    {
      int *a = (int *)omp_alloc(n*sizeof(int), pinned_alloc);
      int *b = (int *)omp_alloc(n*sizeof(int), pinned_alloc);
      int *c = (int *)omp_alloc(n*sizeof(int), pinned_alloc);

      #pragma omp target enter data map(to:a[:n])

      #pragma omp target teams distribute parallel for map(to:b[:n]) map(from:c[:n])
      for(int i = 0; i < n; i++) {
	a[i] = b[i];
	c[i] = i;
      }

      #pragma omp target exit data map(delete:a[:n])

      omp_free(a);
      omp_free(b);
      omp_free(c);
    }
  }

  return 0;
}

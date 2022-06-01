#include <cstdio>
#include <omp.h>

#pragma omp requires unified_shared_memory

int main() {
  int n = 12345;

  int *a = new int[n];
  int *b = new int[n];
  int *da = (int *)omp_target_alloc(n*sizeof(int), /*device=*/omp_get_default_device());

  for(size_t i = 0; i < n; i++)
    b[i] = i;

  // from now on, the runtime will pass 'da' content to the kernels, when 'a' is used in the program
  omp_target_associate_ptr(a, da, n*sizeof(int), /*device_offset=*/0, /*device_num = */0);

  #pragma omp target teams distribute parallel for map(to:b[:n])
  for(size_t i = 0; i < n; i++)
    a[i] = b[i];

  #pragma omp target teams distribute parallel for
  for(size_t i = 0; i < n; i++)
    a[i] += 1;

  #pragma omp target update from(a[:n])

  int err = 0;
  for(size_t i = 0; i < n; i++)
    if (a[i] != i+1) {
      err++;
      printf("Error at %ld, got %d, expected %zu\n", i, a[i], i);
      if (err > 10) return err;
    }
  return err;
}

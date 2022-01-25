#include <cstdio>
#include <omp.h>

#define N 101202

int main() {
  int devnums = 2;
  int devids[] = {0,1};
  int n = N;
  omp_memspace_handle_t managed_memory = omp_get_memory_space(devnums, devids, llvm_omp_target_shared_mem_space);

  omp_allocator_handle_t managed_allocator = omp_init_allocator(managed_memory, 0, {});

  double * arr = (double *)omp_alloc(n*sizeof(double), managed_allocator);

  #pragma omp target teams distribute parallel for map(to:arr)
  for(int i = 0; i < n; i++) {
    arr[i] = i;
  }
  // check
  int err = 0;
  for(int i = 0; i < n; i++)
    if(arr[i] != (double)i) {
      err++;
      printf("Err at %d, expected %lf, got %lf\n", i, (double)i, arr[i]);
      if (err > 10)
	break;
    }

  return err;
}

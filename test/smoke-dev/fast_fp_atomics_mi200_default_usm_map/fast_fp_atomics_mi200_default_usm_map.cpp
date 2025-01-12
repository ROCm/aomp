#include<stdio.h>
#include<omp.h>

// MI200 fast atomic works only on coarse grain memory.
// This test checks the default behavior onf
// USM mapping.

// If default is not enabling coarse grain USM map,
// fast atomic is applied to fine grain memory
// and on MI200, the sum is expected to be zero.

// If default is enabling coarse grain USM map,
// fast atomic is applied to coarse grain memory
// and on MI200, sum should be the same as safe atomics.

// The code below should work both ways, by checking
// whether the stack variable sum is USM mapped to
// coarse grain.

#pragma omp requires unified_shared_memory

int main() {
  double sum = 0.0;
  int n = 10000;
  double valid_sum = (double) n;

  #pragma omp target teams distribute parallel for map(tofrom:sum)
  for(int i = 0; i < n; i++) {
    #pragma omp atomic  hint(AMD_safe_fp_atomics)
    sum+=1.0;
  }

  int err = 0;
  if (sum != (double) n) {
    printf("Error with safe fp atomics, got %lf, expected %lf", sum, (double) n);
    err = 1;
  }

  sum = 0.0;

  #pragma omp target teams distribute parallel for map(tofrom:sum)
  for(int i = 0; i < n; i++) {
    #pragma omp atomic  hint(AMD_fast_fp_atomics)
    sum+=1.0;
  }

  if (!omp_is_coarse_grain_mem_region(&sum, sizeof(double))) {
    valid_sum = 0;
  }
  
  if (sum != valid_sum) {
    printf("Error with fast fp atomics, got %lf, expected %lf", sum, valid_sum);
    err = 1;
  }

  sum = 0.0;

  #pragma omp target teams distribute parallel for map(tofrom:sum)
  for(int i = 0; i < n; i++) {
    #pragma omp atomic  hint(AMD_unsafe_fp_atomics)
    sum+=1.0;
  }

  if (sum != valid_sum) {
    printf("Error with unsafe fp atomics, got %lf, expected %lf", sum, valid_sum);
    err = 1;
  }

  return err;
}

#include<cstddef>

#pragma omp requires unified_shared_memory

void set_to_zero(double *a, size_t n) {
  #pragma omp target teams distribute parallel for map(from:a[:n])
  for(size_t i = 0; i < n; i++)
    a[i] = 0.0;
}

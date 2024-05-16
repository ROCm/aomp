#include <cstdio>
#include <omp.h>

int main() {
  int n = 10000000;
  double *arr = (double *)omp_target_alloc(n*sizeof(double), omp_get_default_device());
  double *red = (double *)omp_target_alloc(1*sizeof(double), omp_get_default_device());

  //init arr on device
  #pragma omp target teams loop
  for(size_t i = 0; i < n; i++)
    arr[i] = i;

  #pragma omp target
  {
    *red = 0.0;
  }

  #pragma omp target teams distribute parallel for
  for(size_t i = 0; i < n; i++) {
    #pragma omp atomic hint(AMD_fast_fp_atomics)
    *red += arr[i];
  }

  double *h = new double[n];
  double hr = 0.0;
  for(size_t i = 0; i < n; i++)
    h[i] = i;
  for(size_t i = 0; i < n; i++)
    hr += h[i];
  
  if (hr != *red) return 1;
  return 0;
}

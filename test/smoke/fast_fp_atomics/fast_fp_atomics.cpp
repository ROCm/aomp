#include<stdio.h>
#include<omp.h>

int main() {
  double sum = 0.0;
  int n = 10000;

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

  if (sum != (double) n) {
    printf("Error with fast fp atomics, got %lf, expected %lf", sum, (double) n);
    err = 1;
  }

  sum = 0.0;

  #pragma omp target teams distribute parallel for map(tofrom:sum)
  for(int i = 0; i < n; i++) {
    #pragma omp atomic  hint(AMD_unsafe_fp_atomics)
    sum+=1.0;
  }

  if (sum != (double) n) {
    printf("Error with unsafe fp atomics, got %lf, expected %lf", sum, (double) n);
    err = 1;
  }

  return err;
}

#include<stdio.h>
#include<omp.h>

int main() {
  double sum = 0.0;
  double x = 107.0;
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

  sum = 0.0;

  #pragma omp target teams distribute parallel for map(tofrom:sum)
  for(int i = 0; i < n; i++) {
    #pragma omp atomic hint(ompx_fast_fp_atomics)
    sum+=1.0;
  }

  if (sum != (double) n) {
    printf("Error with OMPX fast fp atomics, got %lf, expected %lf", sum, (double) n);
    err = 1;
  }

  sum = 0.0;

  #pragma omp target teams distribute parallel for map(tofrom:sum)
  for(int i = 0; i < n; i++) {
    #pragma omp atomic  hint(ompx_unsafe_fp_atomics)
    sum+=1.0;
  }

  if (sum != (double) n) {
    printf("Error with OMPX unsafe fp atomics, got %lf, expected %lf", sum, (double) n);
    err = 1;
  }

  sum = 0.0;

  #pragma omp target teams distribute parallel for map(tofrom:sum)
  for(int i = 0; i < n; i++) {
    #pragma omp atomic  hint(ompx_safe_fp_atomics)
    sum+=1.0;
  }

  if (sum != (double) n) {
    printf("Error with OMPX safe fp atomics, got %lf, expected %lf", sum, (double) n);
    err = 1;
  }

// If the compare clause is present then either statement is:
// cond-expr-stmt, a conditional expression statement that has one of the following forms:
//
// x = expr ordop x ? expr : x;
// x = x ordop expr ? expr : x;
//
// or cond-update-stmt, a conditional update statement that has one of the following forms:
//
// if(expr ordop x) { x = expr; }
// if(x ordop expr) { x = expr; }
//

  x = 107.0;

  #pragma omp target teams distribute parallel for map(tofrom:x)
  for(int i = 0; i < n; i++) {
    #pragma omp atomic compare hint(ompx_fast_fp_atomics)
    x = i < x ? i : x; // MIN
  }

  if (x != 0) {
    printf("Error with OMPX fast fp atomics, got %f, expected %d\n", x, 0);
    err = 1;
  }

  x = 107.0;

  #pragma omp target teams distribute parallel for map(tofrom:x)
  for(int i = 0; i < n; i++) {
    #pragma omp atomic compare hint(ompx_fast_fp_atomics)
    x = i > x ? i : x; // MAX
  }

  if (x != n - 1) {
    printf("Error with OMPX fast fp atomics, got %f, expected %d\n", x, n-1);
    err = 1;
  }

  x = 107.0;

  #pragma omp target teams distribute parallel for map(tofrom:x)
  for(int i = 0; i < n; i++) {
    #pragma omp atomic compare hint(ompx_fast_fp_atomics)
    x = x > i ? i : x; // MIN
  }

  if (x != 0) {
    printf("Error with OMPX fast fp atomics, got %f, expected %d\n", x, 0);
    err = 1;
  }

  x = 107.0;

  #pragma omp target teams distribute parallel for map(tofrom:x)
  for(int i = 0; i < n; i++) {
    #pragma omp atomic compare hint(ompx_fast_fp_atomics)
    x = x < i ? i : x; // MAX
  }

  if (x != n - 1) {
    printf("Error with OMPX fast fp atomics, got %f, expected %d\n", x, n-1);
    err = 1;
  }

  x = 107.0;

  #pragma omp target teams distribute parallel for map(tofrom:x)
  for(int i = 0; i < n; i++) {
    #pragma omp atomic compare hint(ompx_fast_fp_atomics)
    if(i < x) { x = i; } // MIN
  }

  if (x != 0) {
    printf("Error with OMPX fast fp atomics, got %f, expected %d\n", x, 0);
    err = 1;
  }

  x = 107.0;

  #pragma omp target teams distribute parallel for map(tofrom:x)
  for(int i = 0; i < n; i++) {
    #pragma omp atomic compare hint(ompx_fast_fp_atomics)
    if(i > x) { x = i; } // MAX
  }

  if (x != n - 1) {
    printf("Error with OMPX fast fp atomics, got %f, expected %d\n", x, n-1);
    err = 1;
  }

  x = 107.0;

  #pragma omp target teams distribute parallel for map(tofrom:x)
  for(int i = 0; i < n; i++) {
    #pragma omp atomic compare hint(ompx_fast_fp_atomics)
    if(x > i) { x = i; } // MIN
  }

  if (x != 0) {
    printf("Error with OMPX fast fp atomics, got %f, expected %d\n", x, 0);
    err = 1;
  }

  x = 107.0;

  #pragma omp target teams distribute parallel for map(tofrom:x)
  for(int i = 0; i < n; i++) {
    #pragma omp atomic compare hint(ompx_fast_fp_atomics)
    if(x < i) { x = i; } // MAX
  }

  if (x != n - 1) {
    printf("Error with OMPX fast fp atomics, got %f, expected %d\n", x, n-1);
    err = 1;
  }

  return err;
}

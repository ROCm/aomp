#include <stdio.h>
#include <stdlib.h>

// 22 gets no errors after 500 iterations.
//#define DIM 22

// With 23, we start seeing failures. Not always but definitely repeatable.
#define DIM 23

// With 32, we fail at one executiono or two at the most.
// #define DIM 32
#define N (DIM*DIM)

int main() {
  int a[N], b[N], c[N];
  int errors = 0;
  // Data Inititalize
  for (int i = 0; i < N; i++) {
    a[i] = 2*i;  // Even
    b[i] = 2*i + 1;  // Odd
    c[i] = 0;
  }

  #pragma omp target teams distribute parallel for map(to: a, b) map(from: c) collapse(2)
  // #pragma omp target data map(to: a,b) map(from: c)
  // #pragma omp target teams distribute parallel for collapse(2)
  for (int i = 0; i < DIM; i++) {
    for (int j = (i*DIM); j < (i*DIM + DIM); j++) {
      c[j] = a[j] + b[j];
    }
  }

  for (int i = 0; i < N; i++) {
    if (c[i] != a[i] + b[i]) {
      printf("Fail with c[%d] is %d\n", i, c[i]);
      ++errors;
    }
  }
  return errors;

}

/// CHECK: SGN:4

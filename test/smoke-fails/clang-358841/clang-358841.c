#include <assert.h>
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define N 100

int main() {

  double array[N];
  int errors = 0;
  
  // Array initialization
  for (int i = 0; i < N; ++i) {
    array[i] = 0.99;
  }
  // This is intentional
  int c99_zero = FP_ZERO;
  
#pragma omp target map(tofrom: array[0:N]) 
  for (int i = 0; i < N; ++i) {
    array[i] = pow((double)i,2.0);
  }

  for (int i = 0; i < N; ++i) {
    if ((array[i] - pow((double)i,2)) > 0.000009) {
      fprintf(stderr, "wrong %d %f \n", i, array[i]);
      errors++;
    }
  }
  return errors;
}

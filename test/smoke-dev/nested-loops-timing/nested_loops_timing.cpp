#include <stdio.h>
#include <omp.h>

// Sizes representative of the Spec benchmark
#define OUTER 15000
#define INNER_SMALL 10
#define INNER_LARGE 60000

int check_results(double *A, double *B, double *x) {
  int error = 0;

  // Check results are correct:
  for(int i=0; i<OUTER * INNER_SMALL; i++) {
    if (A[i] != 0.0) {
      printf("FAIL: A[%d] = %f but expected %f\n", i, A[i], 0.0);
      error += 1;
      break;
    }
  }

  for(int i=0; i<OUTER; i++) {
    for(int j=0; j<INNER_LARGE; j++) {
      if (B[i * INNER_LARGE + j] != x[j]) {
        printf("FAIL: B[%d] = %f but expected %f\n", i * INNER_LARGE + j, B[i * INNER_LARGE + j], x[i]);
        error += 1;
        break;
      }
    }
    if (error > 0)
      break;
  }

  return error;
}

int main() {
  int error = 0;

  // An array for the small inner loop:
  double *A = (double *)malloc(sizeof(double) * OUTER * INNER_SMALL);

  // An array for the large inner loop:
  double *B = (double *)malloc(sizeof(double) * OUTER * INNER_LARGE);

  // The array to be copied on every line of the matrix B:
  double *x = (double *)malloc(sizeof(double) * INNER_LARGE);

  for(int i=0; i<INNER_LARGE; i++) {
    x[i] = i;
  }

  // Map data to GPU:
#pragma omp target enter data map(to:x[:INNER_LARGE])
#pragma omp target enter data map(alloc:A[:OUTER * INNER_SMALL], B[:OUTER * INNER_LARGE])

  // Run original loop:

  double t0 = omp_get_wtime();
  #pragma omp target teams distribute parallel for
  for(int k=0; k<OUTER; k++) {
    #pragma omp simd
    for(int i=0; i<INNER_SMALL; i++)
      A[k*INNER_SMALL + i] = 0.0;
    #pragma omp simd
    for(int i=0; i<INNER_LARGE; i++)
      B[k*INNER_LARGE + i] = x[i];
  }
  double original = omp_get_wtime() - t0;

#pragma omp target update from(A[:OUTER * INNER_SMALL], B[:OUTER * INNER_LARGE])

  error += check_results(A, B, x);

  // Run loop nest with loop directives:

  t0 = omp_get_wtime();
  #pragma omp target teams loop
  for(int k=0; k<OUTER; k++) {
    #pragma omp loop
    for(int i=0; i<INNER_SMALL; i++)
      A[k*INNER_SMALL + i] = 0.0;
    #pragma omp loop
    for(int i=0; i<INNER_LARGE; i++)
      B[k*INNER_LARGE + i] = x[i];
  }
  double loop_nest_with_loops = omp_get_wtime() - t0;

#pragma omp target update from(A[:OUTER * INNER_SMALL], B[:OUTER * INNER_LARGE])

  error += check_results(A, B, x);

  // Run collapsed split loops:

  t0 = omp_get_wtime();
  #pragma omp target teams distribute parallel for collapse(2)
  for(int k=0; k<OUTER; k++) {
    for(int i=0; i<INNER_SMALL; i++)
      A[k*INNER_SMALL + i] = 0.0;
  }
  double split_loop_1 = omp_get_wtime() - t0;

  t0 = omp_get_wtime();
  #pragma omp target teams distribute parallel for collapse(2)
  for(int k=0; k<OUTER; k++) {
    for(int i=0; i<INNER_LARGE; i++)
      B[k*INNER_LARGE + i] = x[i];
  }
  double split_loop_2 = omp_get_wtime() - t0;

#pragma omp target update from(A[:OUTER * INNER_SMALL], B[:OUTER * INNER_LARGE])

  error += check_results(A, B, x);

#pragma omp target exit data map(delete:x[:INNER_LARGE])
#pragma omp target exit data map(delete:A[:OUTER * INNER_SMALL], B[:OUTER * INNER_LARGE])

  printf("Original loop nest runtime            = %f\n", original);
  printf("Loop nest with loop directives        = %f\n", loop_nest_with_loops);
  printf("Split and collapsed loop nest runtime = %f (%f + %f)\n", split_loop_1 + split_loop_2, split_loop_1, split_loop_2);

  if (error == 0) {
    printf("Success\n");
  }

  free(A);
  free(B);
  free(x);

  return error;
}

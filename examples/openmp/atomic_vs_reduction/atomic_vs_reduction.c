/*
 * atomic_vs_reduction.c
 * 
 * This program demonstrates the performance difference between atomic operations 
 * and reductions in OpenMP. It calculates the sum of integers from 0 to N-1 using:
 * 1. Atomic operations with a hint for fast floating-point atomics (if supported).
 * 2. Atomic operations without any hint.
 * 3. Reduction operations.
 * 
 * The program compares the execution time and correctness of each approach.
 */

#include <stdio.h>
#include <omp.h>

int main() {
  // Initialize return code and problem size
  int main_rc = 0;
  int N = 5001; // Number of integers to sum
  float expect = (float)(((float)N - 1) * (float)N) / 2.0; // Expected sum

#if defined(__LLVM_COMPILER_NAME_IS_CLANG__)
  // Skip fast FP atomic hint if not using AMD Compiler
  printf("Skipping fast FP atomic hint because this is not AMD Compiler\n");
#else
  // Test atomic operations with fast FP atomic hint
  float a = 0.0;
  double t0 = omp_get_wtime();
#pragma omp target teams distribute parallel for map(tofrom : a)
  for (int ii = 0; ii < N; ++ii) {
#pragma omp atomic hint(AMD_fast_fp_atomics)
    a += (float)ii; // Atomic addition with hint
  }
  double t1 = omp_get_wtime() - t0;

  // Check correctness and print results
  if (a == expect) {
    printf("Success atomic with hint (fast FP atomic) sum of %d integers is: %f in \t\t%f secs\n", N, a, t1);
  } else {
    printf("FAIL ATOMIC SUM N:%d result: %f != expect: %f \n", N, a, expect);
    main_rc = 1;
  }
#endif

  // Test atomic operations without any hint
  float casa = 0.0;
  double t_cas0 = omp_get_wtime();
#pragma omp target teams distribute parallel for map(tofrom : casa)
  for (int ii = 0; ii < N; ++ii) {
#pragma omp atomic
    casa += (float)ii; // Atomic addition without hint
  }
  double t_cas1 = omp_get_wtime() - t_cas0;

  // Check correctness and print results
  if (casa == expect) {
    printf("Success atomic without hint (cas loop) sum of %d integers is: %f in \t\t%f secs\n", N, casa, t_cas1);
  } else {
    printf("FAIL ATOMIC SUM N:%d result: %f != expect: %f \n", N, casa, expect);
    main_rc = 1;
  }

  // Test reduction operations
  float ra = 0.0;
  double t2 = omp_get_wtime();
#pragma omp target teams distribute parallel for reduction(+ : ra)
  for (int ii = 0; ii < N; ++ii) {
    ra += (float)ii; // Reduction addition
  }
  double t3 = omp_get_wtime() - t2;

  // Check correctness and print results
  if (ra == expect) {
    printf("Success reduction sum of %d integers is: %f in \t\t\t\t\t%f secs\n", N, ra, t3);
  } else {
    printf("FAIL REDUCTION SUM N:%d result: %f != expect: %f \n", N, ra, expect);
    main_rc = 1;
  }

  // Return the overall result
  return main_rc;
}

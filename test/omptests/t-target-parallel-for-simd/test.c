
#include <stdio.h>
#include <omp.h>

#include "../utilities/check.h"
#include "../utilities/utilities.h"

#define TRIALS (1)

#define N (1024*3)

#define INIT() INIT_LOOP(N, {C[i] = 1; D[i] = i; E[i] = -i+1;})

#define ZERO(X) ZERO_ARRAY(N, X) 

#define DUMP_SUCCESS9() { \
  DUMP_SUCCESS(gpu_threads-max_threads); \
  DUMP_SUCCESS(gpu_threads-max_threads); \
  DUMP_SUCCESS(gpu_threads-max_threads); \
  DUMP_SUCCESS(gpu_threads-max_threads); \
  DUMP_SUCCESS(gpu_threads-max_threads); \
  DUMP_SUCCESS(gpu_threads-max_threads); \
  DUMP_SUCCESS(gpu_threads-max_threads); \
  DUMP_SUCCESS(gpu_threads-max_threads); \
  DUMP_SUCCESS(gpu_threads-max_threads); \
}

//
// FIXME:
//   Add support for 'shared', 'lastprivate'
//

int main(void) {
  check_offloading();

  double A[N], B[N], C[N], D[N], E[N];
  double S[N];
  double p[2];

  int cpuExec = 0;
  #pragma omp target map(tofrom: cpuExec)
  {
    cpuExec = omp_is_initial_device();
  }
  int gpu_threads = 224;
  int cpu_threads = 32;
  int max_threads = cpuExec ? cpu_threads : gpu_threads;

  INIT();

  //
  // Test: proc_bind clause
  //
  #undef TARGET_PARALLEL_FOR_CLAUSES
  #define TARGET_PARALLEL_FOR_CLAUSES proc_bind(master)
  #include "tpf_defines.h"
  for (int t = 0; t <= max_threads; t++) {
    int threads[1]; threads[0] = t;
    TARGET_PARALLEL_FOR1(
    {
      S[0] = 0;
      for (int i = 0; i < N; i++) {
        A[i] = B[i] = 0;
      }
    },
    for (int i = 0; i < N; i++) { \
      A[i] += C[i] + D[i]; \
      B[i] += D[i] + E[i]; \
    },
    {
      double tmp = 0;
      for (int i = 0; i < N; i++) {
        tmp += A[i] + B[i] - 1;
      }
      S[0] += tmp;
    },
    VERIFY(0, 1, S[0], N/2*(N+1)))

    TARGET_PARALLEL_FOR2(
    {
      S[0] = 0;
      for (int i = 0; i < N; i++) {
        A[i] = B[i] = 0;
      }
    },
    for (int i = 0; i < N; i++) { \
      A[i] += C[i] + D[i]; \
      B[i] += D[i] + E[i]; \
    },
    {
      double tmp = 0;
      for (int i = 0; i < N; i++) {
        tmp += A[i] + B[i] - 1;
      }
      S[0] += tmp;
    },
    VERIFY(0, 1, S[0], N/2*(N+1)))

    TARGET_PARALLEL_FOR3(
    {
      S[0] = 0;
      for (int i = 0; i < N; i++) {
        A[i] = B[i] = 0;
      }
    },
    for (int i = 0; i < N; i++) { \
      A[i] += C[i] + D[i]; \
      B[i] += D[i] + E[i]; \
    },
    {
      double tmp = 0;
      for (int i = 0; i < N; i++) {
        tmp += A[i] + B[i] - 1;
      }
      S[0] += tmp;
    },
    VERIFY(0, 1, S[0], N/2*(N+1)))

    TARGET_PARALLEL_FOR4(
    {
      S[0] = 0;
      for (int i = 0; i < N; i++) {
        A[i] = B[i] = 0;
      }
    },
    for (int i = 0; i < N; i++) { \
      A[i] += C[i] + D[i]; \
      B[i] += D[i] + E[i]; \
    },
    {
      double tmp = 0;
      for (int i = 0; i < N; i++) {
        tmp += A[i] + B[i] - 1;
      }
      S[0] += tmp;
    },
    VERIFY(0, 1, S[0], N/2*(N+1)))

    TARGET_PARALLEL_FOR5(
    {
      S[0] = 0;
      for (int i = 0; i < N; i++) {
        A[i] = B[i] = 0;
      }
    },
    for (int i = 0; i < N; i++) { \
      A[i] += C[i] + D[i]; \
      B[i] += D[i] + E[i]; \
    },
    {
      double tmp = 0;
      for (int i = 0; i < N; i++) {
        tmp += A[i] + B[i] - 1;
      }
      S[0] += tmp;
    },
    VERIFY(0, 1, S[0], N/2*(N+1)))

    TARGET_PARALLEL_FOR6(
    {
      S[0] = 0;
      for (int i = 0; i < N; i++) {
        A[i] = B[i] = 0;
      }
    },
    for (int i = 0; i < N; i++) { \
      A[i] += C[i] + D[i]; \
      B[i] += D[i] + E[i]; \
    },
    {
      double tmp = 0;
      for (int i = 0; i < N; i++) {
        tmp += A[i] + B[i] - 1;
      }
      S[0] += tmp;
    },
    VERIFY(0, 1, S[0], N/2*(N+1)))

    TARGET_PARALLEL_FOR7(
    {
      S[0] = 0;
      for (int i = 0; i < N; i++) {
        A[i] = B[i] = 0;
      }
    },
    for (int i = 0; i < N; i++) { \
      A[i] += C[i] + D[i]; \
      B[i] += D[i] + E[i]; \
    },
    {
      double tmp = 0;
      for (int i = 0; i < N; i++) {
        tmp += A[i] + B[i] - 1;
      }
      S[0] += tmp;
    },
    VERIFY(0, 1, S[0], N/2*(N+1)))

    TARGET_PARALLEL_FOR8(
    {
      S[0] = 0;
      for (int i = 0; i < N; i++) {
        A[i] = B[i] = 0;
      }
    },
    for (int i = 0; i < N; i++) { \
      A[i] += C[i] + D[i]; \
      B[i] += D[i] + E[i]; \
    },
    {
      double tmp = 0;
      for (int i = 0; i < N; i++) {
        tmp += A[i] + B[i] - 1;
      }
      S[0] += tmp;
    },
    VERIFY(0, 1, S[0], N/2*(N+1)))

    TARGET_PARALLEL_FOR9(
    {
      S[0] = 0;
      for (int i = 0; i < N; i++) {
        A[i] = B[i] = 0;
      }
    },
    for (int i = 0; i < N; i++) { \
      A[i] += C[i] + D[i]; \
      B[i] += D[i] + E[i]; \
    },
    {
      double tmp = 0;
      for (int i = 0; i < N; i++) {
        tmp += A[i] + B[i] - 1;
      }
      S[0] += tmp;
    },
    VERIFY(0, 1, S[0], N/2*(N+1)))
  }
  DUMP_SUCCESS9()

  #undef TARGET_PARALLEL_FOR_CLAUSES
  #define TARGET_PARALLEL_FOR_CLAUSES proc_bind(close)
  #include "tpf_defines.h"
  for (int t = 0; t <= max_threads; t++) {
    int threads[1]; threads[0] = t;
    TARGET_PARALLEL_FOR1(
    {
      S[0] = 0;
      for (int i = 0; i < N; i++) {
        A[i] = B[i] = 0;
      }
    },
    for (int i = 0; i < N; i++) { \
      A[i] += C[i] + D[i]; \
      B[i] += D[i] + E[i]; \
    },
    {
      double tmp = 0;
      for (int i = 0; i < N; i++) {
        tmp += A[i] + B[i] - 1;
      }
      S[0] += tmp;
    },
    VERIFY(0, 1, S[0], N/2*(N+1)))

    TARGET_PARALLEL_FOR2(
    {
      S[0] = 0;
      for (int i = 0; i < N; i++) {
        A[i] = B[i] = 0;
      }
    },
    for (int i = 0; i < N; i++) { \
      A[i] += C[i] + D[i]; \
      B[i] += D[i] + E[i]; \
    },
    {
      double tmp = 0;
      for (int i = 0; i < N; i++) {
        tmp += A[i] + B[i] - 1;
      }
      S[0] += tmp;
    },
    VERIFY(0, 1, S[0], N/2*(N+1)))

    TARGET_PARALLEL_FOR3(
    {
      S[0] = 0;
      for (int i = 0; i < N; i++) {
        A[i] = B[i] = 0;
      }
    },
    for (int i = 0; i < N; i++) { \
      A[i] += C[i] + D[i]; \
      B[i] += D[i] + E[i]; \
    },
    {
      double tmp = 0;
      for (int i = 0; i < N; i++) {
        tmp += A[i] + B[i] - 1;
      }
      S[0] += tmp;
    },
    VERIFY(0, 1, S[0], N/2*(N+1)))

    TARGET_PARALLEL_FOR4(
    {
      S[0] = 0;
      for (int i = 0; i < N; i++) {
        A[i] = B[i] = 0;
      }
    },
    for (int i = 0; i < N; i++) { \
      A[i] += C[i] + D[i]; \
      B[i] += D[i] + E[i]; \
    },
    {
      double tmp = 0;
      for (int i = 0; i < N; i++) {
        tmp += A[i] + B[i] - 1;
      }
      S[0] += tmp;
    },
    VERIFY(0, 1, S[0], N/2*(N+1)))

    TARGET_PARALLEL_FOR5(
    {
      S[0] = 0;
      for (int i = 0; i < N; i++) {
        A[i] = B[i] = 0;
      }
    },
    for (int i = 0; i < N; i++) { \
      A[i] += C[i] + D[i]; \
      B[i] += D[i] + E[i]; \
    },
    {
      double tmp = 0;
      for (int i = 0; i < N; i++) {
        tmp += A[i] + B[i] - 1;
      }
      S[0] += tmp;
    },
    VERIFY(0, 1, S[0], N/2*(N+1)))

    TARGET_PARALLEL_FOR6(
    {
      S[0] = 0;
      for (int i = 0; i < N; i++) {
        A[i] = B[i] = 0;
      }
    },
    for (int i = 0; i < N; i++) { \
      A[i] += C[i] + D[i]; \
      B[i] += D[i] + E[i]; \
    },
    {
      double tmp = 0;
      for (int i = 0; i < N; i++) {
        tmp += A[i] + B[i] - 1;
      }
      S[0] += tmp;
    },
    VERIFY(0, 1, S[0], N/2*(N+1)))

    TARGET_PARALLEL_FOR7(
    {
      S[0] = 0;
      for (int i = 0; i < N; i++) {
        A[i] = B[i] = 0;
      }
    },
    for (int i = 0; i < N; i++) { \
      A[i] += C[i] + D[i]; \
      B[i] += D[i] + E[i]; \
    },
    {
      double tmp = 0;
      for (int i = 0; i < N; i++) {
        tmp += A[i] + B[i] - 1;
      }
      S[0] += tmp;
    },
    VERIFY(0, 1, S[0], N/2*(N+1)))

    TARGET_PARALLEL_FOR8(
    {
      S[0] = 0;
      for (int i = 0; i < N; i++) {
        A[i] = B[i] = 0;
      }
    },
    for (int i = 0; i < N; i++) { \
      A[i] += C[i] + D[i]; \
      B[i] += D[i] + E[i]; \
    },
    {
      double tmp = 0;
      for (int i = 0; i < N; i++) {
        tmp += A[i] + B[i] - 1;
      }
      S[0] += tmp;
    },
    VERIFY(0, 1, S[0], N/2*(N+1)))

    TARGET_PARALLEL_FOR9(
    {
      S[0] = 0;
      for (int i = 0; i < N; i++) {
        A[i] = B[i] = 0;
      }
    },
    for (int i = 0; i < N; i++) { \
      A[i] += C[i] + D[i]; \
      B[i] += D[i] + E[i]; \
    },
    {
      double tmp = 0;
      for (int i = 0; i < N; i++) {
        tmp += A[i] + B[i] - 1;
      }
      S[0] += tmp;
    },
    VERIFY(0, 1, S[0], N/2*(N+1)))
  }
  DUMP_SUCCESS9()

  #undef TARGET_PARALLEL_FOR_CLAUSES
  #define TARGET_PARALLEL_FOR_CLAUSES proc_bind(spread)
  #include "tpf_defines.h"
  for (int t = 0; t <= max_threads; t++) {
    int threads[1]; threads[0] = t;
    TARGET_PARALLEL_FOR1(
    {
      S[0] = 0;
      for (int i = 0; i < N; i++) {
        A[i] = B[i] = 0;
      }
    },
    for (int i = 0; i < N; i++) { \
      A[i] += C[i] + D[i]; \
      B[i] += D[i] + E[i]; \
    },
    {
      double tmp = 0;
      for (int i = 0; i < N; i++) {
        tmp += A[i] + B[i] - 1;
      }
      S[0] += tmp;
    },
    VERIFY(0, 1, S[0], N/2*(N+1)))

    TARGET_PARALLEL_FOR2(
    {
      S[0] = 0;
      for (int i = 0; i < N; i++) {
        A[i] = B[i] = 0;
      }
    },
    for (int i = 0; i < N; i++) { \
      A[i] += C[i] + D[i]; \
      B[i] += D[i] + E[i]; \
    },
    {
      double tmp = 0;
      for (int i = 0; i < N; i++) {
        tmp += A[i] + B[i] - 1;
      }
      S[0] += tmp;
    },
    VERIFY(0, 1, S[0], N/2*(N+1)))

    TARGET_PARALLEL_FOR3(
    {
      S[0] = 0;
      for (int i = 0; i < N; i++) {
        A[i] = B[i] = 0;
      }
    },
    for (int i = 0; i < N; i++) { \
      A[i] += C[i] + D[i]; \
      B[i] += D[i] + E[i]; \
    },
    {
      double tmp = 0;
      for (int i = 0; i < N; i++) {
        tmp += A[i] + B[i] - 1;
      }
      S[0] += tmp;
    },
    VERIFY(0, 1, S[0], N/2*(N+1)))

    TARGET_PARALLEL_FOR4(
    {
      S[0] = 0;
      for (int i = 0; i < N; i++) {
        A[i] = B[i] = 0;
      }
    },
    for (int i = 0; i < N; i++) { \
      A[i] += C[i] + D[i]; \
      B[i] += D[i] + E[i]; \
    },
    {
      double tmp = 0;
      for (int i = 0; i < N; i++) {
        tmp += A[i] + B[i] - 1;
      }
      S[0] += tmp;
    },
    VERIFY(0, 1, S[0], N/2*(N+1)))

    TARGET_PARALLEL_FOR5(
    {
      S[0] = 0;
      for (int i = 0; i < N; i++) {
        A[i] = B[i] = 0;
      }
    },
    for (int i = 0; i < N; i++) { \
      A[i] += C[i] + D[i]; \
      B[i] += D[i] + E[i]; \
    },
    {
      double tmp = 0;
      for (int i = 0; i < N; i++) {
        tmp += A[i] + B[i] - 1;
      }
      S[0] += tmp;
    },
    VERIFY(0, 1, S[0], N/2*(N+1)))

    TARGET_PARALLEL_FOR6(
    {
      S[0] = 0;
      for (int i = 0; i < N; i++) {
        A[i] = B[i] = 0;
      }
    },
    for (int i = 0; i < N; i++) { \
      A[i] += C[i] + D[i]; \
      B[i] += D[i] + E[i]; \
    },
    {
      double tmp = 0;
      for (int i = 0; i < N; i++) {
        tmp += A[i] + B[i] - 1;
      }
      S[0] += tmp;
    },
    VERIFY(0, 1, S[0], N/2*(N+1)))

    TARGET_PARALLEL_FOR7(
    {
      S[0] = 0;
      for (int i = 0; i < N; i++) {
        A[i] = B[i] = 0;
      }
    },
    for (int i = 0; i < N; i++) { \
      A[i] += C[i] + D[i]; \
      B[i] += D[i] + E[i]; \
    },
    {
      double tmp = 0;
      for (int i = 0; i < N; i++) {
        tmp += A[i] + B[i] - 1;
      }
      S[0] += tmp;
    },
    VERIFY(0, 1, S[0], N/2*(N+1)))

    TARGET_PARALLEL_FOR8(
    {
      S[0] = 0;
      for (int i = 0; i < N; i++) {
        A[i] = B[i] = 0;
      }
    },
    for (int i = 0; i < N; i++) { \
      A[i] += C[i] + D[i]; \
      B[i] += D[i] + E[i]; \
    },
    {
      double tmp = 0;
      for (int i = 0; i < N; i++) {
        tmp += A[i] + B[i] - 1;
      }
      S[0] += tmp;
    },
    VERIFY(0, 1, S[0], N/2*(N+1)))

    TARGET_PARALLEL_FOR9(
    {
      S[0] = 0;
      for (int i = 0; i < N; i++) {
        A[i] = B[i] = 0;
      }
    },
    for (int i = 0; i < N; i++) { \
      A[i] += C[i] + D[i]; \
      B[i] += D[i] + E[i]; \
    },
    {
      double tmp = 0;
      for (int i = 0; i < N; i++) {
        tmp += A[i] + B[i] - 1;
      }
      S[0] += tmp;
    },
    VERIFY(0, 1, S[0], N/2*(N+1)))
  }
  DUMP_SUCCESS9()

  //
  // Test: private, shared clauses on omp target parallel for.
  //
  #undef TARGET_PARALLEL_FOR_CLAUSES
  #define TARGET_PARALLEL_FOR_CLAUSES private(p,q) shared(A,B,C,D,E)
  #include "tpf_defines.h"
  // FIXME: shared(a) where 'a' is an implicitly mapped scalar does not work.
  // FIXME: shared(A) private(A) does not generate correct results.
  for (int t = 0; t <= max_threads; t++) {
    int threads[1]; threads[0] = t;
    TARGET_PARALLEL_FOR1(
      double p = 2; \
      double q = 4; \
      S[0] = 0; \
      for (int i = 0; i < N; i++) { \
        A[i] = B[i] = 0; \
      }
    ,
    for (int i = 0; i < N; i++) { \
      p = C[i] + D[i]; \
      q = D[i] + E[i]; \
      A[i] += p; \
      B[i] += q; \
    },
    {
      double tmp = p + q;
      for (int i = 0; i < N; i++) {
        tmp += A[i] + B[i] - 1;
      }
      S[0] += tmp;
    },
    VERIFY(0, 1, S[0], 6 + N/2*(N+1)))

    TARGET_PARALLEL_FOR2(
      double p = 2; \
      double q = 4; \
      S[0] = 0; \
      for (int i = 0; i < N; i++) { \
        A[i] = B[i] = 0; \
      }
    ,
    for (int i = 0; i < N; i++) { \
      p = C[i] + D[i]; \
      q = D[i] + E[i]; \
      A[i] += p; \
      B[i] += q; \
    },
    {
      double tmp = p + q;
      for (int i = 0; i < N; i++) {
        tmp += A[i] + B[i] - 1;
      }
      S[0] += tmp;
    },
    VERIFY(0, 1, S[0], 6 + N/2*(N+1)))

    TARGET_PARALLEL_FOR3(
      double p = 2; \
      double q = 4; \
      S[0] = 0; \
      for (int i = 0; i < N; i++) { \
        A[i] = B[i] = 0; \
      }
    ,
    for (int i = 0; i < N; i++) { \
      p = C[i] + D[i]; \
      q = D[i] + E[i]; \
      A[i] += p; \
      B[i] += q; \
    },
    {
      double tmp = p + q;
      for (int i = 0; i < N; i++) {
        tmp += A[i] + B[i] - 1;
      }
      S[0] += tmp;
    },
    VERIFY(0, 1, S[0], 6 + N/2*(N+1)))

    TARGET_PARALLEL_FOR4(
      double p = 2; \
      double q = 4; \
      S[0] = 0; \
      for (int i = 0; i < N; i++) { \
        A[i] = B[i] = 0; \
      }
    ,
    for (int i = 0; i < N; i++) { \
      p = C[i] + D[i]; \
      q = D[i] + E[i]; \
      A[i] += p; \
      B[i] += q; \
    },
    {
      double tmp = p + q;
      for (int i = 0; i < N; i++) {
        tmp += A[i] + B[i] - 1;
      }
      S[0] += tmp;
    },
    VERIFY(0, 1, S[0], 6 + N/2*(N+1)))

    TARGET_PARALLEL_FOR5(
      double p = 2; \
      double q = 4; \
      S[0] = 0; \
      for (int i = 0; i < N; i++) { \
        A[i] = B[i] = 0; \
      }
    ,
    for (int i = 0; i < N; i++) { \
      p = C[i] + D[i]; \
      q = D[i] + E[i]; \
      A[i] += p; \
      B[i] += q; \
    },
    {
      double tmp = p + q;
      for (int i = 0; i < N; i++) {
        tmp += A[i] + B[i] - 1;
      }
      S[0] += tmp;
    },
    VERIFY(0, 1, S[0], 6 + N/2*(N+1)))

    TARGET_PARALLEL_FOR6(
      double p = 2; \
      double q = 4; \
      S[0] = 0; \
      for (int i = 0; i < N; i++) { \
        A[i] = B[i] = 0; \
      }
    ,
    for (int i = 0; i < N; i++) { \
      p = C[i] + D[i]; \
      q = D[i] + E[i]; \
      A[i] += p; \
      B[i] += q; \
    },
    {
      double tmp = p + q;
      for (int i = 0; i < N; i++) {
        tmp += A[i] + B[i] - 1;
      }
      S[0] += tmp;
    },
    VERIFY(0, 1, S[0], 6 + N/2*(N+1)))

    TARGET_PARALLEL_FOR7(
      double p = 2; \
      double q = 4; \
      S[0] = 0; \
      for (int i = 0; i < N; i++) { \
        A[i] = B[i] = 0; \
      }
    ,
    for (int i = 0; i < N; i++) { \
      p = C[i] + D[i]; \
      q = D[i] + E[i]; \
      A[i] += p; \
      B[i] += q; \
    },
    {
      double tmp = p + q;
      for (int i = 0; i < N; i++) {
        tmp += A[i] + B[i] - 1;
      }
      S[0] += tmp;
    },
    VERIFY(0, 1, S[0], 6 + N/2*(N+1)))

    TARGET_PARALLEL_FOR8(
      double p = 2; \
      double q = 4; \
      S[0] = 0; \
      for (int i = 0; i < N; i++) { \
        A[i] = B[i] = 0; \
      }
    ,
    for (int i = 0; i < N; i++) { \
      p = C[i] + D[i]; \
      q = D[i] + E[i]; \
      A[i] += p; \
      B[i] += q; \
    },
    {
      double tmp = p + q;
      for (int i = 0; i < N; i++) {
        tmp += A[i] + B[i] - 1;
      }
      S[0] += tmp;
    },
    VERIFY(0, 1, S[0], 6 + N/2*(N+1)))

    TARGET_PARALLEL_FOR9(
      double p = 2; \
      double q = 4; \
      S[0] = 0; \
      for (int i = 0; i < N; i++) { \
        A[i] = B[i] = 0; \
      }
    ,
    for (int i = 0; i < N; i++) { \
      p = C[i] + D[i]; \
      q = D[i] + E[i]; \
      A[i] += p; \
      B[i] += q; \
    },
    {
      double tmp = p + q;
      for (int i = 0; i < N; i++) {
        tmp += A[i] + B[i] - 1;
      }
      S[0] += tmp;
    },
    VERIFY(0, 1, S[0], 6 + N/2*(N+1)))
  }
  DUMP_SUCCESS9()

  //
  // Test: firstprivate clause on omp target parallel for.
  //
  #undef TARGET_PARALLEL_FOR_CLAUSES
  #define TARGET_PARALLEL_FOR_CLAUSES firstprivate(p,q)
  #include "tpf_defines.h"
  for (int t = 0; t <= max_threads; t++) {
    int threads[1]; threads[0] = t;
    TARGET_PARALLEL_FOR1(
      double p = -4; \
      double q = 4; \
      S[0] = 0; \
      for (int i = 0; i < N; i++) { \
        A[i] = B[i] = 0; \
      }
    ,
    for (int i = 0; i < N; i++) { \
      A[i] += C[i] + D[i] + p; \
      B[i] += D[i] + E[i] + q; \
      if (i == N-1) { \
        p += 6; \
        q += 9; \
      } \
    },
    {
      double tmp = p + q;
      for (int i = 0; i < N; i++) {
        tmp += A[i] + B[i] - 1;
      }
      S[0] += tmp;
    },
    VERIFY(0, 1, S[0], N/2*(N+1)))

    TARGET_PARALLEL_FOR2(
      double p = -4; \
      double q = 4; \
      S[0] = 0; \
      for (int i = 0; i < N; i++) { \
        A[i] = B[i] = 0; \
      }
    ,
    for (int i = 0; i < N; i++) { \
      A[i] += C[i] + D[i] + p; \
      B[i] += D[i] + E[i] + q; \
      if (i == N-1) { \
        p += 6; \
        q += 9; \
      } \
    },
    {
      double tmp = p + q;
      for (int i = 0; i < N; i++) {
        tmp += A[i] + B[i] - 1;
      }
      S[0] += tmp;
    },
    VERIFY(0, 1, S[0], N/2*(N+1)))

    TARGET_PARALLEL_FOR3(
      double p = -4; \
      double q = 4; \
      S[0] = 0; \
      for (int i = 0; i < N; i++) { \
        A[i] = B[i] = 0; \
      }
    ,
    for (int i = 0; i < N; i++) { \
      A[i] += C[i] + D[i] + p; \
      B[i] += D[i] + E[i] + q; \
      if (i == N-1) { \
        p += 6; \
        q += 9; \
      } \
    },
    {
      double tmp = p + q;
      for (int i = 0; i < N; i++) {
        tmp += A[i] + B[i] - 1;
      }
      S[0] += tmp;
    },
    VERIFY(0, 1, S[0], N/2*(N+1)))

    TARGET_PARALLEL_FOR4(
      double p = -4; \
      double q = 4; \
      S[0] = 0; \
      for (int i = 0; i < N; i++) { \
        A[i] = B[i] = 0; \
      }
    ,
    for (int i = 0; i < N; i++) { \
      A[i] += C[i] + D[i] + p; \
      B[i] += D[i] + E[i] + q; \
      if (i == N-1) { \
        p += 6; \
        q += 9; \
      } \
    },
    {
      double tmp = p + q;
      for (int i = 0; i < N; i++) {
        tmp += A[i] + B[i] - 1;
      }
      S[0] += tmp;
    },
    VERIFY(0, 1, S[0], N/2*(N+1)))

    TARGET_PARALLEL_FOR5(
      double p = -4; \
      double q = 4; \
      S[0] = 0; \
      for (int i = 0; i < N; i++) { \
        A[i] = B[i] = 0; \
      }
    ,
    for (int i = 0; i < N; i++) { \
      A[i] += C[i] + D[i] + p; \
      B[i] += D[i] + E[i] + q; \
      if (i == N-1) { \
        p += 6; \
        q += 9; \
      } \
    },
    {
      double tmp = p + q;
      for (int i = 0; i < N; i++) {
        tmp += A[i] + B[i] - 1;
      }
      S[0] += tmp;
    },
    VERIFY(0, 1, S[0], N/2*(N+1)))

    TARGET_PARALLEL_FOR6(
      double p = -4; \
      double q = 4; \
      S[0] = 0; \
      for (int i = 0; i < N; i++) { \
        A[i] = B[i] = 0; \
      }
    ,
    for (int i = 0; i < N; i++) { \
      A[i] += C[i] + D[i] + p; \
      B[i] += D[i] + E[i] + q; \
      if (i == N-1) { \
        p += 6; \
        q += 9; \
      } \
    },
    {
      double tmp = p + q;
      for (int i = 0; i < N; i++) {
        tmp += A[i] + B[i] - 1;
      }
      S[0] += tmp;
    },
    VERIFY(0, 1, S[0], N/2*(N+1)))

    TARGET_PARALLEL_FOR7(
      double p = -4; \
      double q = 4; \
      S[0] = 0; \
      for (int i = 0; i < N; i++) { \
        A[i] = B[i] = 0; \
      }
    ,
    for (int i = 0; i < N; i++) { \
      A[i] += C[i] + D[i] + p; \
      B[i] += D[i] + E[i] + q; \
      if (i == N-1) { \
        p += 6; \
        q += 9; \
      } \
    },
    {
      double tmp = p + q;
      for (int i = 0; i < N; i++) {
        tmp += A[i] + B[i] - 1;
      }
      S[0] += tmp;
    },
    VERIFY(0, 1, S[0], N/2*(N+1)))

    TARGET_PARALLEL_FOR8(
      double p = -4; \
      double q = 4; \
      S[0] = 0; \
      for (int i = 0; i < N; i++) { \
        A[i] = B[i] = 0; \
      }
    ,
    for (int i = 0; i < N; i++) { \
      A[i] += C[i] + D[i] + p; \
      B[i] += D[i] + E[i] + q; \
      if (i == N-1) { \
        p += 6; \
        q += 9; \
      } \
    },
    {
      double tmp = p + q;
      for (int i = 0; i < N; i++) {
        tmp += A[i] + B[i] - 1;
      }
      S[0] += tmp;
    },
    VERIFY(0, 1, S[0], N/2*(N+1)))

    TARGET_PARALLEL_FOR9(
      double p = -4; \
      double q = 4; \
      S[0] = 0; \
      for (int i = 0; i < N; i++) { \
        A[i] = B[i] = 0; \
      }
    ,
    for (int i = 0; i < N; i++) { \
      A[i] += C[i] + D[i] + p; \
      B[i] += D[i] + E[i] + q; \
      if (i == N-1) { \
        p += 6; \
        q += 9; \
      } \
    },
    {
      double tmp = p + q;
      for (int i = 0; i < N; i++) {
        tmp += A[i] + B[i] - 1;
      }
      S[0] += tmp;
    },
    VERIFY(0, 1, S[0], N/2*(N+1)))
  }
  DUMP_SUCCESS9()

#if 0
  //
  // Test: lastprivate clause on omp target parallel for.
  //
  #undef TARGET_PARALLEL_FOR_CLAUSES
  #define TARGET_PARALLEL_FOR_CLAUSES lastprivate(q)
  #include "tpf_defines.h"
  // FIXME: modify to t=1 and in tpf_defines.h to use host after bug fix.
  // FIXME: variable is not private.
  for (int t = 2; t <= max_threads; t++) {
    int threads[1]; threads[0] = t;
    TARGET_PARALLEL_FOR1(
      double p[1]; \
      double q[1]; \
      S[0] = 0; \
      for (int i = 0; i < N; i++) { \
        A[i] = B[i] = 0; \
      }
    ,
    for (int i = 0; i < N; i++) { \
      p[0] = C[i] + D[i]; \
      q[0] = D[i] + E[i]; \
      A[i] = p[0]; \
      B[i] = q[0]; \
    },
    {
      double tmp = p[0] + q[0];
      for (int i = 0; i < N; i++) {
        tmp += A[i] + B[i] - 1;
      }
      S[0] += tmp;
    },
    VERIFY(0, 1, S[0], N+1+ N/2*(N+1)))
  }

  FIXME: private of non-scalar does not work.
  //
  // Test: private clause on omp parallel for.
  //
  #undef PARALLEL_FOR_CLAUSES
  #define PARALLEL_FOR_CLAUSES private(p)
  #include "tpf_defines.h"
  for (int t = 0; t <= 224; t++) {
    int threads[1]; threads[0] = t;
    PARALLEL_FOR(
      p[0] = 2; p[1] = 4; \
      S[0] = 0; \
      for (int i = 0; i < N; i++) { \
        A[i] = B[i] = 0; \
      }
    ,
    for (int i = 0; i < N; i++) { \
      p[0] = C[i] + D[i]; \
      p[1] = D[i] + E[i]; \
      A[i] += p[0]; \
      B[i] += p[1]; \
    }
    ,
    {
      double tmp = p[0] + p[1];
      for (int i = 0; i < N; i++) {
        tmp += A[i] + B[i];
      }
      S[0] += tmp;
    },
    VERIFY(0, 1, S[0], 6 + SUMS * (N/2*(N+1))))
  }

  FIXME: private of non-scalar does not work.
  //
  // Test: firstprivate clause on omp parallel for.
  //
  #undef PARALLEL_FOR_CLAUSES
  #define PARALLEL_FOR_CLAUSES firstprivate(p)
  #include "tpf_defines.h"
  for (int t = 0; t <= 224; t++) {
    int threads[1]; threads[0] = t;
    PARALLEL_FOR(
      p[0] = -4; p[1] = 4; \
      S[0] = 0; \
      for (int i = 0; i < N; i++) { \
        A[i] = B[i] = 0; \
      }
    ,
    for (int i = 0; i < N; i++) { \
      A[i] += C[i] + D[i] + p[0]; \
      B[i] += D[i] + E[i] + p[1]; \
      if (i == N-1) { \
        p[0] += 6; \
        p[1] += 9; \
      } \
    }
    ,
    {
      double tmp = p[0] + p[1];
      for (int i = 0; i < N; i++) {
        tmp += A[i] + B[i];
      }
      S[0] += tmp;
    },
    VERIFY(0, 1, S[0], SUMS * (N/2*(N+1))))
  }
#endif

  //
  // Test: collapse clause on omp target parallel for.
  //
  #undef TARGET_PARALLEL_FOR_CLAUSES
  #define TARGET_PARALLEL_FOR_CLAUSES collapse(2)
  #include "tpf_defines.h"
  for (int t = 0; t <= max_threads; t++) {
    int threads[1]; threads[0] = t;
    TARGET_PARALLEL_FOR1(
      S[0] = 0; \
      for (int i = 0; i < N; i++) { \
        A[i] = B[i] = 0; \
      }
    ,
    for (int i = 0; i < 1024; i++) { \
      for (int j = 0; j < 3; j++) { \
        A[i*3+j] += C[i*3+j] + D[i*3+j]; \
        B[i*3+j] += D[i*3+j] + E[i*3+j]; \
      } \
    }
    ,
    {
      double tmp = 0;
      for (int i = 0; i < N; i++) {
        tmp += A[i] + B[i] - 1;
      }
      S[0] += tmp;
    },
    VERIFY(0, 1, S[0], (N/2*(N+1))))

    TARGET_PARALLEL_FOR2(
      S[0] = 0; \
      for (int i = 0; i < N; i++) { \
        A[i] = B[i] = 0; \
      }
    ,
    for (int i = 0; i < 1024; i++) { \
      for (int j = 0; j < 3; j++) { \
        A[i*3+j] += C[i*3+j] + D[i*3+j]; \
        B[i*3+j] += D[i*3+j] + E[i*3+j]; \
      } \
    }
    ,
    {
      double tmp = 0;
      for (int i = 0; i < N; i++) {
        tmp += A[i] + B[i] - 1;
      }
      S[0] += tmp;
    },
    VERIFY(0, 1, S[0], (N/2*(N+1))))

    TARGET_PARALLEL_FOR3(
      S[0] = 0; \
      for (int i = 0; i < N; i++) { \
        A[i] = B[i] = 0; \
      }
    ,
    for (int i = 0; i < 1024; i++) { \
      for (int j = 0; j < 3; j++) { \
        A[i*3+j] += C[i*3+j] + D[i*3+j]; \
        B[i*3+j] += D[i*3+j] + E[i*3+j]; \
      } \
    }
    ,
    {
      double tmp = 0;
      for (int i = 0; i < N; i++) {
        tmp += A[i] + B[i] - 1;
      }
      S[0] += tmp;
    },
    VERIFY(0, 1, S[0], (N/2*(N+1))))

    TARGET_PARALLEL_FOR4(
      S[0] = 0; \
      for (int i = 0; i < N; i++) { \
        A[i] = B[i] = 0; \
      }
    ,
    for (int i = 0; i < 1024; i++) { \
      for (int j = 0; j < 3; j++) { \
        A[i*3+j] += C[i*3+j] + D[i*3+j]; \
        B[i*3+j] += D[i*3+j] + E[i*3+j]; \
      } \
    }
    ,
    {
      double tmp = 0;
      for (int i = 0; i < N; i++) {
        tmp += A[i] + B[i] - 1;
      }
      S[0] += tmp;
    },
    VERIFY(0, 1, S[0], (N/2*(N+1))))

    TARGET_PARALLEL_FOR5(
      S[0] = 0; \
      for (int i = 0; i < N; i++) { \
        A[i] = B[i] = 0; \
      }
    ,
    for (int i = 0; i < 1024; i++) { \
      for (int j = 0; j < 3; j++) { \
        A[i*3+j] += C[i*3+j] + D[i*3+j]; \
        B[i*3+j] += D[i*3+j] + E[i*3+j]; \
      } \
    }
    ,
    {
      double tmp = 0;
      for (int i = 0; i < N; i++) {
        tmp += A[i] + B[i] - 1;
      }
      S[0] += tmp;
    },
    VERIFY(0, 1, S[0], (N/2*(N+1))))

    TARGET_PARALLEL_FOR6(
      S[0] = 0; \
      for (int i = 0; i < N; i++) { \
        A[i] = B[i] = 0; \
      }
    ,
    for (int i = 0; i < 1024; i++) { \
      for (int j = 0; j < 3; j++) { \
        A[i*3+j] += C[i*3+j] + D[i*3+j]; \
        B[i*3+j] += D[i*3+j] + E[i*3+j]; \
      } \
    }
    ,
    {
      double tmp = 0;
      for (int i = 0; i < N; i++) {
        tmp += A[i] + B[i] - 1;
      }
      S[0] += tmp;
    },
    VERIFY(0, 1, S[0], (N/2*(N+1))))

    TARGET_PARALLEL_FOR7(
      S[0] = 0; \
      for (int i = 0; i < N; i++) { \
        A[i] = B[i] = 0; \
      }
    ,
    for (int i = 0; i < 1024; i++) { \
      for (int j = 0; j < 3; j++) { \
        A[i*3+j] += C[i*3+j] + D[i*3+j]; \
        B[i*3+j] += D[i*3+j] + E[i*3+j]; \
      } \
    }
    ,
    {
      double tmp = 0;
      for (int i = 0; i < N; i++) {
        tmp += A[i] + B[i] - 1;
      }
      S[0] += tmp;
    },
    VERIFY(0, 1, S[0], (N/2*(N+1))))

    TARGET_PARALLEL_FOR8(
      S[0] = 0; \
      for (int i = 0; i < N; i++) { \
        A[i] = B[i] = 0; \
      }
    ,
    for (int i = 0; i < 1024; i++) { \
      for (int j = 0; j < 3; j++) { \
        A[i*3+j] += C[i*3+j] + D[i*3+j]; \
        B[i*3+j] += D[i*3+j] + E[i*3+j]; \
      } \
    }
    ,
    {
      double tmp = 0;
      for (int i = 0; i < N; i++) {
        tmp += A[i] + B[i] - 1;
      }
      S[0] += tmp;
    },
    VERIFY(0, 1, S[0], (N/2*(N+1))))

    TARGET_PARALLEL_FOR9(
      S[0] = 0; \
      for (int i = 0; i < N; i++) { \
        A[i] = B[i] = 0; \
      }
    ,
    for (int i = 0; i < 1024; i++) { \
      for (int j = 0; j < 3; j++) { \
        A[i*3+j] += C[i*3+j] + D[i*3+j]; \
        B[i*3+j] += D[i*3+j] + E[i*3+j]; \
      } \
    }
    ,
    {
      double tmp = 0;
      for (int i = 0; i < N; i++) {
        tmp += A[i] + B[i] - 1;
      }
      S[0] += tmp;
    },
    VERIFY(0, 1, S[0], (N/2*(N+1))))
  }
  DUMP_SUCCESS9()

  //
  // Test: Ensure coalesced scheduling on GPU.
  //
  if (cpuExec == 0) {
    #undef TARGET_PARALLEL_FOR_CLAUSES
    #define TARGET_PARALLEL_FOR_CLAUSES
    #include "tpf_defines.h"
    int threads[1]; threads[0] = 33;
    TARGET_PARALLEL_FOR1(
      S[0] = 0; \
      for (int i = 0; i < 99; i++) { \
        A[i] = 0; \
      } \
    ,
      for (int i = 0; i < 99; i++) { \
        A[i] += i - omp_get_thread_num(); \
      }
    ,
    {
      double tmp = 0;
      for (int i = 0; i < 99; i++) {
        tmp += A[i];
      }
      S[0] = tmp;
    },
    VERIFY(0, 1, S[0], (33*33 + 66*33) ))

    TARGET_PARALLEL_FOR2(
      S[0] = 0; \
      for (int i = 0; i < 99; i++) { \
        A[i] = 0; \
      } \
    ,
      for (int i = 0; i < 99; i++) { \
        A[i] += i - omp_get_thread_num(); \
      }
    ,
    {
      double tmp = 0;
      for (int i = 0; i < 99; i++) {
        tmp += A[i];
      }
      S[0] = tmp;
    },
    VERIFY(0, 1, S[0], (33*33 + 66*33) ))

    TARGET_PARALLEL_FOR7(
      S[0] = 0; \
      for (int i = 0; i < 99; i++) { \
        A[i] = 0; \
      } \
    ,
      for (int i = 0; i < 99; i++) { \
        A[i] += i - omp_get_thread_num(); \
      }
    ,
    {
      double tmp = 0;
      for (int i = 0; i < 99; i++) {
        tmp += A[i];
      }
      S[0] = tmp;
    },
    VERIFY(0, 1, S[0], (33*33 + 66*33) ))
  } else {
    DUMP_SUCCESS(3);
  }

  return 0;
}


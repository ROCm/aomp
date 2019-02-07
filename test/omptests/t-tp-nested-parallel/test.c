
#include <stdio.h>
#include <omp.h>

#include "../utilities/check.h"
#include "../utilities/utilities.h"

#define TRIALS (1)

#define N (1024*3)
#define M (65)

#define INIT() INIT_LOOP(N, {C[i] = 1; D[i] = i; E[i] = -i;})

#define ZERO(X) ZERO_ARRAY(N, X) 

double A[M][N], B[M][N], C[N], D[N], E[N];
double S[M];
double p[2];

int main(void) {
  check_offloading();

  INIT();

  int cpuExec = 0;
  #pragma omp target map(tofrom: cpuExec)
  {
    cpuExec = omp_is_initial_device();
  }

  //
  // Test: proc_bind clause
  //
  #undef NESTED_PARALLEL_FOR_CLAUSES
  #define NESTED_PARALLEL_FOR_CLAUSES proc_bind(master)
  #include "defines.h"
  for (int t = 1; t <= 64; t++) {
    int threads[1]; threads[0] = t-1;
    NESTED_PARALLEL_FOR(
    int tid = omp_get_thread_num(); \
    S[tid] = 0; \
    for (int i = 0; i < N; i++) { \
      A[tid][i] = B[tid][i] = 0; \
    },
    for (int i = 0; i < N; i++) { \
      A[tid][i] += C[i] + D[i]; \
      B[tid][i] += D[i] + E[i]; \
    },
    {
      double tmp = 0;
      for (int i = 0; i < N; i++) {
        tmp += A[tid][i] + B[tid][i];
      }
      S[tid] += tmp;
    },
    VERIFY(0, t, S[i], (double) SUMS * (N/2*(N+1))))
  }

  #undef NESTED_PARALLEL_FOR_CLAUSES
  #define NESTED_PARALLEL_FOR_CLAUSES proc_bind(close)
  #include "defines.h"
  for (int t = 1; t <= 64; t++) {
    int threads[1]; threads[0] = t-1;
    NESTED_PARALLEL_FOR(
    int tid = omp_get_thread_num(); \
    S[tid] = 0; \
    for (int i = 0; i < N; i++) { \
      A[tid][i] = B[tid][i] = 0; \
    },
    for (int i = 0; i < N; i++) { \
      A[tid][i] += C[i] + D[i]; \
      B[tid][i] += D[i] + E[i]; \
    },
    {
      double tmp = 0;
      for (int i = 0; i < N; i++) {
        tmp += A[tid][i] + B[tid][i];
      }
      S[tid] += tmp;
    },
    VERIFY(0, t, S[i], (double) SUMS * (N/2*(N+1))))
  }

  #undef NESTED_PARALLEL_FOR_CLAUSES
  #define NESTED_PARALLEL_FOR_CLAUSES proc_bind(spread)
  #include "defines.h"
  for (int t = 1; t <= 64; t++) {
    int threads[1]; threads[0] = t-1;
    NESTED_PARALLEL_FOR(
    int tid = omp_get_thread_num(); \
    S[tid] = 0; \
    for (int i = 0; i < N; i++) { \
      A[tid][i] = B[tid][i] = 0; \
    },
    for (int i = 0; i < N; i++) { \
      A[tid][i] += C[i] + D[i]; \
      B[tid][i] += D[i] + E[i]; \
    },
    {
      double tmp = 0;
      for (int i = 0; i < N; i++) {
        tmp += A[tid][i] + B[tid][i];
      }
      S[tid] += tmp;
    },
    VERIFY(0, t, S[i], (double) SUMS * (N/2*(N+1))))
  }

  //
  // Test: private, shared clauses on omp target parallel for with nested parallel.
  //
  #undef NESTED_PARALLEL_FOR_CLAUSES
  #define NESTED_PARALLEL_FOR_CLAUSES private(p,q) shared(A,B,C,D,E)
  #include "defines.h"
  for (int t = 1; t <= 64; t++) {
    int threads[1]; threads[0] = t;
    NESTED_PARALLEL_FOR(
      double p = 2; \
      double q = 4; \
      int tid = omp_get_thread_num(); \
      S[tid] = 0; \
      for (int i = 0; i < N; i++) { \
        A[tid][i] = B[tid][i] = 0; \
      },
    for (int i = 0; i < N; i++) { \
      p = C[i] + D[i]; \
      q = D[i] + E[i]; \
      A[tid][i] += p; \
      B[tid][i] += q; \
    }
    ,
    {
      double tmp = p + q;
      for (int i = 0; i < N; i++) {
        tmp += A[tid][i] + B[tid][i];
      }
      S[tid] += tmp;
    },
    VERIFY(0, t, S[i], (double) 6 + SUMS * (N/2*(N+1))))
  }

  //
  // Test: firstprivate clause on omp target parallel for with nested parallel.
  //
  #undef NESTED_PARALLEL_FOR_CLAUSES
  #define NESTED_PARALLEL_FOR_CLAUSES firstprivate(p,q)
  #include "defines.h"
  for (int t = 1; t <= 64; t++) {
    int threads[1]; threads[0] = t;
    NESTED_PARALLEL_FOR(
      double p = -4; \
      double q = 4; \
      int tid = omp_get_thread_num(); \
      S[tid] = 0; \
      for (int i = 0; i < N; i++) { \
        A[tid][i] = B[tid][i] = 0; \
      },
    for (int i = 0; i < N; i++) { \
      A[tid][i] += C[i] + D[i] + p; \
      B[tid][i] += D[i] + E[i] + q; \
      if (i == N-1) { \
        p += 6; \
        q += 9; \
      } \
    }
    ,
    {
      double tmp = p + q;
      for (int i = 0; i < N; i++) {
        tmp += A[tid][i] + B[tid][i];
      }
      S[tid] += tmp;
    },
    VERIFY(0, t, S[i], (double) SUMS * (N/2*(N+1))))
  }

  //
  // Test: lastprivate clause on omp target parallel for with nested parallel.
  //
  for (int t = 1; t <= 64; t++) {
    int threads[1]; threads[0] = t;
    TESTD("omp target parallel num_threads(t)", {
      double q0[1];
      double q1[1];
      double q2[1];
      double q3[1];
      int tid = omp_get_thread_num();
      S[tid] = 0;
      for (int i = 0; i < N; i++) {
        A[tid][i] = B[tid][i] = 0;
      }
      _Pragma("omp parallel for lastprivate(q0) if(threads[0] > 1) num_threads(threads[0])")
      for (int i = 0; i < N; i++) {
        q0[0] = C[i] + D[i];
        A[tid][i] += q0[0];
      }
      _Pragma("omp parallel for schedule(auto) lastprivate(q1) if(threads[0] > 1) num_threads(threads[0])")
      for (int i = 0; i < N; i++) {
        q1[0] = C[i] + D[i];
        A[tid][i] += q1[0];
      }
      _Pragma("omp parallel for schedule(static) lastprivate(q2) if(threads[0] > 1) num_threads(threads[0])")
      for (int i = 0; i < N; i++) {
        q2[0] = D[i] + E[i];
        B[tid][i] += q2[0];
      }
      _Pragma("omp parallel for schedule(static,9) lastprivate(q3) if(threads[0] > 1) num_threads(threads[0])")
      for (int i = 0; i < N; i++) {
        q3[0] = D[i] + E[i];
        B[tid][i] += q3[0];
      }
      double tmp = q0[0] + q1[0] + q2[0] + q3[0];
      for (int i = 0; i < N; i++) {
        tmp += A[tid][i] + B[tid][i];
      }
      S[tid] += tmp;
    }, VERIFY(0, t, S[i], (double) 2 * (N + (N/2*(N+1))) ));
  }

  //
  // Test: private clause on omp target parallel for with nested parallel.
  //
  #undef NESTED_PARALLEL_FOR_CLAUSES
  #define NESTED_PARALLEL_FOR_CLAUSES private(p)
  #include "defines.h"
  for (int t = 1; t <= 64; t++) {
    int threads[1]; threads[0] = t;
    NESTED_PARALLEL_FOR(
      double p[2]; \
      p[0] = 2; p[1] = 4; \
      int tid = omp_get_thread_num(); \
      S[tid] = 0; \
      for (int i = 0; i < N; i++) { \
        A[tid][i] = B[tid][i] = 0; \
      }
    ,
    for (int i = 0; i < N; i++) { \
      p[0] = C[i] + D[i]; \
      p[1] = D[i] + E[i]; \
      A[tid][i] += p[0]; \
      B[tid][i] += p[1]; \
    }
    ,
    {
      double tmp = p[0] + p[1];
      for (int i = 0; i < N; i++) {
        tmp += A[tid][i] + B[tid][i];
      }
      S[tid] += tmp;
    },
    VERIFY(0, t, S[i], (double) 6 + SUMS * (N/2*(N+1))))
  }

  //
  // Test: firstprivate clause on omp target parallel for with nested parallel.
  //
  #undef NESTED_PARALLEL_FOR_CLAUSES
  #define NESTED_PARALLEL_FOR_CLAUSES firstprivate(p)
  #include "defines.h"
  for (int t = 1; t <= 64; t++) {
    int threads[1]; threads[0] = t;
    NESTED_PARALLEL_FOR(
      double p[2]; \
      p[0] = -4; p[1] = 4; \
      int tid = omp_get_thread_num(); \
      S[tid] = 0; \
      for (int i = 0; i < N; i++) { \
        A[tid][i] = B[tid][i] = 0; \
      }
    ,
    for (int i = 0; i < N; i++) { \
      A[tid][i] += C[i] + D[i] + p[0]; \
      B[tid][i] += D[i] + E[i] + p[1]; \
      if (i == N-1) { \
        p[0] += 6; \
        p[1] += 9; \
      } \
    }
    ,
    {
      double tmp = p[0] + p[1];
      for (int i = 0; i < N; i++) {
        tmp += A[tid][i] + B[tid][i];
      }
      S[tid] += tmp;
    },
    VERIFY(0, t, S[i], (double) SUMS * (N/2*(N+1))))
  }

  //
  // Test: collapse clause on omp target parallel for with nested parallel.
  //
  #undef NESTED_PARALLEL_FOR_CLAUSES
  #define NESTED_PARALLEL_FOR_CLAUSES collapse(2)
  #include "defines.h"
  for (int t = 1; t <= 64; t++) {
    int threads[1]; threads[0] = t;
    NESTED_PARALLEL_FOR(
      int tid = omp_get_thread_num(); \
      S[tid] = 0; \
      for (int i = 0; i < N; i++) { \
        A[tid][i] = B[tid][i] = 0; \
      }
    ,
    for (int i = 0; i < 1024; i++) { \
      for (int j = 0; j < 3; j++) { \
        A[tid][i*3+j] += C[i*3+j] + D[i*3+j]; \
        B[tid][i*3+j] += D[i*3+j] + E[i*3+j]; \
      } \
    }
    ,
    {
      double tmp = 0;
      for (int i = 0; i < N; i++) {
        tmp += A[tid][i] + B[tid][i];
      }
      S[tid] += tmp;
    },
    VERIFY(0, t, S[i], (double) SUMS * (N/2*(N+1))))
  }

  //
  // Test: ordered clause on omp target parallel for with nested parallel.
  //
  #undef NESTED_PARALLEL_FOR_CLAUSES
  #define NESTED_PARALLEL_FOR_CLAUSES ordered
  #include "defines.h"
  for (int t = 1; t <= 64; t += 64) {
    int threads[1]; threads[0] = t;
    NESTED_PARALLEL_FOR(
      int tid = omp_get_thread_num(); \
      S[tid] = 0; \
    ,
    for (int i = 0; i < N; i++) { \
      _Pragma("omp ordered") \
      S[tid] += C[i] + D[i]; \
    }
    ,
    {
    },
    VERIFY(0, t, S[i], (double) SUMS * (N/2*(N+1))))
  }

  //
  // Test: Ensure coalesced scheduling on GPU.
  //
  if (!cpuExec) {
    TESTD("omp target parallel num_threads(32)", {
      int tid = omp_get_thread_num();
      S[tid] = 0;
      for (int i = 0; i < 96; i++) {
        A[tid][i] = 0;
      }
      _Pragma("omp parallel for num_threads(32)")
      for (int i = 0; i < 96; i++) {
        A[tid][i] += i - omp_get_thread_num();
      }
      _Pragma("omp parallel for schedule(auto) num_threads(32)")
      for (int i = 0; i < 96; i++) {
        A[tid][i] += i - omp_get_thread_num();
      } 
      _Pragma("omp parallel for schedule(static,1) num_threads(32)")
      for (int i = 0; i < 96; i++) {
        A[tid][i] += i - omp_get_thread_num();
      }
      double tmp = 0;
      for (int i = 0; i < 96; i++) {
        tmp += A[tid][i];
      }
      S[tid] = tmp;
    }, VERIFY(0, 32, S[i], (double) 3 * (32*32 + 64*32) ));
  } else {
    DUMP_SUCCESS(1);
  }

  return 0;
}

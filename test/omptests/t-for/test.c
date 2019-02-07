
#include <stdio.h>
#include <omp.h>

#include "../utilities/check.h"
#include "../utilities/utilities.h"

#define TRIALS (1)

#define N (1024*3)

#define INIT() INIT_LOOP(N, {C[i] = 1; D[i] = i; E[i] = -i;})

#define ZERO(X) ZERO_ARRAY(N, X)

int main(void) {
  check_offloading();

  double A[N], B[N], C[N], D[N], E[N];
  double S[N];
  double p[2];

  INIT();

  long cpuExec = 0;
  #pragma omp target map(tofrom: cpuExec)
  {
    cpuExec = omp_is_initial_device();
  }
  int max_threads = 224;

  #undef FOR_CLAUSES
  #define FOR_CLAUSES
  #include "defines.h"
  for (int t = 0; t <= max_threads; t++) {
    int threads[1]; threads[0] = t;
    PARALLEL(
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
        tmp += A[i] + B[i];
      }
      S[0] += tmp;
    },
    VERIFY(0, 1, S[0], SUMS * (N/2*(N+1))))
  }

  //
  // Test: private clause on omp for.
  //
  #undef FOR_CLAUSES
  #define FOR_CLAUSES private(p,q)
  #include "defines.h"
  for (int t = 0; t <= max_threads; t++) {
    int threads[1]; threads[0] = t;
    PARALLEL(
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
    }
    ,
    {
      double tmp = p + q;
      for (int i = 0; i < N; i++) {
        tmp += A[i] + B[i];
      }
      S[0] += tmp;
    },
    VERIFY(0, 1, S[0], 6 + SUMS * (N/2*(N+1))))
  }

  //
  // Test: firstprivate clause on omp for.
  //
  #undef FOR_CLAUSES
  #define FOR_CLAUSES firstprivate(p,q)
  #include "defines.h"
  for (int t = 0; t <= max_threads; t++) {
    int threads[1]; threads[0] = t;
    PARALLEL(
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
    }
    ,
    {
      double tmp = p + q;
      for (int i = 0; i < N; i++) {
        tmp += A[i] + B[i];
      }
      S[0] += tmp;
    },
    VERIFY(0, 1, S[0], SUMS * (N/2*(N+1))))
  }

  //
  // Test: lastprivate clause on omp for.
  //
  double q0[1], q1[1], q2[1], q3[1], q4[1], q5[1], q6[1], q7[1], q8[1], q9[1];
  for (int t = 0; t <= max_threads; t++) {
    int threads[1]; threads[0] = t;
    TEST({
      S[0] = 0;
      for (int i = 0; i < N; i++) {
        A[i] = B[i] = 0;
      }
    _Pragma("omp parallel if(threads[0] > 1) num_threads(threads[0])")
      {
        _Pragma("omp for lastprivate(q0)")
        for (int i = 0; i < N; i++) {
          q0[0] = C[i] + D[i];
          A[i] += q0[0];
        }
        _Pragma("omp for schedule(auto) lastprivate(q1)")
        for (int i = 0; i < N; i++) {
          q1[0] = D[i] + E[i];
          B[i] += q1[0];
        }
        _Pragma("omp for schedule(dynamic) lastprivate(q2)")
        for (int i = 0; i < N; i++) {
          q2[0] = C[i] + D[i];
          A[i] += q2[0];
        }
        _Pragma("omp for schedule(guided) lastprivate(q3)")
        for (int i = 0; i < N; i++) {
          q3[0] = D[i] + E[i];
          B[i] += q3[0];
        }
        _Pragma("omp for schedule(runtime) lastprivate(q4)")
        for (int i = 0; i < N; i++) {
          q4[0] = C[i] + D[i];
          A[i] += q4[0];
        }
        _Pragma("omp for schedule(static) lastprivate(q5)")
        for (int i = 0; i < N; i++) {
          q5[0] = D[i] + E[i];
          B[i] += q5[0];
        }
        _Pragma("omp for schedule(static,1) lastprivate(q6)")
        for (int i = 0; i < N; i++) {
          q6[0] = C[i] + D[i];
          A[i] += q6[0];
        }
        _Pragma("omp for schedule(static,9) lastprivate(q7)")
        for (int i = 0; i < N; i++) {
          q7[0] = D[i] + E[i];
          B[i] += q7[0];
        }
        _Pragma("omp for schedule(static,13) lastprivate(q8)")
        for (int i = 0; i < N; i++) {
          q8[0] = C[i] + D[i];
          A[i] += q8[0];
        }
        _Pragma("omp for schedule(static,30000) lastprivate(q9)")
        for (int i = 0; i < N; i++) {
          q9[0] = D[i] + E[i];
          B[i] += q9[0];
        }
      }
      double tmp = q0[0] + q1[0] + q2[0] + q3[0] + q4[0] + \
                   q5[0] + q6[0] + q7[0] + q8[0] + q9[0];
      for (int i = 0; i < N; i++) {
        tmp += A[i] + B[i];
      }
      S[0] += tmp;
    }, VERIFY(0, 1, S[0], 5 * (N + (N/2*(N+1))) ));
  }

  //
  // Test: private clause on omp for.
  //
  #undef FOR_CLAUSES
  #define FOR_CLAUSES private(p)
  #include "defines.h"
  for (int t = 0; t <= max_threads; t++) {
    int threads[1]; threads[0] = t;
    PARALLEL(
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

  //
  // Test: firstprivate clause on omp for.
  //
  #undef FOR_CLAUSES
  #define FOR_CLAUSES firstprivate(p)
  #include "defines.h"
  for (int t = 0; t <= max_threads; t++) {
    int threads[1]; threads[0] = t;
    PARALLEL(
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

  //
  // Test: collapse clause on omp for.
  //
  #undef FOR_CLAUSES
  #define FOR_CLAUSES collapse(2)
  #include "defines.h"
  for (int t = 0; t <= max_threads; t++) {
    int threads[1]; threads[0] = t;
    PARALLEL(
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
        tmp += A[i] + B[i];
      }
      S[0] += tmp;
    },
    VERIFY(0, 1, S[0], SUMS * (N/2*(N+1))))
  }

  //
  // Test: ordered clause on omp for.
  //
  #undef FOR_CLAUSES
  #define FOR_CLAUSES ordered
  #include "defines.h"
  for (int t = 0; t <= max_threads; t += max_threads) {
    int threads[1]; threads[0] = t;
    PARALLEL(
      S[0] = 0; \
    ,
    for (int i = 0; i < N; i++) { \
      _Pragma("omp ordered") \
      S[0] += C[i] + D[i]; \
    }
    ,
    {
    },
    VERIFY(0, 1, S[0], SUMS * (N/2*(N+1))))
  }

  //
  // Test: nowait clause on omp for.
  // FIXME: Not sure how to test for correctness.
  //
  for (int t = 0; t <= max_threads; t++) {
    int threads[1]; threads[0] = t;
    TEST({
      S[0] = 0;
      for (int i = 0; i < N; i++) {
        A[i] = B[i] = 0;
      }
    _Pragma("omp parallel if(threads[0] > 1) num_threads(threads[0])")
      {
        _Pragma("omp for nowait schedule(static,1)")
        for (int i = 0; i < N; i++) {
          A[i] = C[i] + D[i];
        }
        _Pragma("omp for nowait schedule(static,1)")
        for (int i = 0; i < N; i++) {
          B[i] = A[i] + D[i] + E[i];
        }
        _Pragma("omp barrier")
        if (omp_get_thread_num() == 0) {
          double tmp = 0;
          for (int i = 0; i < N; i++) {
            tmp += B[i];
          }
          S[0] += tmp;
        }
      }
    }, VERIFY(0, 1, S[0], (N/2*(N+1)) ));
  }

  //
  // Test: Ensure coalesced scheduling on GPU.
  //
  if (!cpuExec) {
    TESTD("pragma omp target teams num_teams(1) thread_limit(33)", {
      S[0] = 0;
      for (int i = 0; i < 99; i++) {
        A[i] = 0;
      }
    _Pragma("omp parallel num_threads(33)")
      {
        _Pragma("omp for")
        for (int i = 0; i < 99; i++) {
          A[i] += i - omp_get_thread_num();
        }
        _Pragma("omp for schedule(auto)")
        for (int i = 0; i < 99; i++) {
          A[i] += i - omp_get_thread_num();
        }
        _Pragma("omp for schedule(static,1)")
        for (int i = 0; i < 99; i++) {
          A[i] += i - omp_get_thread_num();
        }
      }
      double tmp = 0;
      for (int i = 0; i < 99; i++) {
        tmp += A[i];
      }
      S[0] = tmp;
    }, VERIFY(0, 1, S[0], 3 * (33*33 + 66*33) ));
  } else {
    DUMP_SUCCESS(1);
  }

  //
  // Test: Ensure that we have barriers after dynamic, guided,
  // and ordered schedules, even with a nowait clause since the
  // NVPTX runtime doesn't currently support concurrent execution
  // of these constructs.
  // FIXME: Not sure how to test for correctness at runtime.
  //
  if (!cpuExec) {
    TEST({
      for (int i = 0; i < N; i++) {
        A[i] = 0;
      }
    _Pragma("omp parallel")
      {
        _Pragma("omp for nowait schedule(guided)")
        for (int i = 0; i < N; i++) {
          A[i] += C[i] + D[i];
        }
        _Pragma("omp for nowait schedule(dynamic)")
        for (int i = 0; i < N; i++) {
          A[i] += D[i] + E[i];
        }
        _Pragma("omp for nowait ordered")
        for (int i = 0; i < N; i++) {
          A[i] += C[i] + D[i];
        }
      }
    }, VERIFY(0, N, A[i], 2*i+2) );
  } else {
    DUMP_SUCCESS(1);
  }

  //
  // Test: Linear clause on target
  //
  if (!cpuExec) {
    int l = 0;
    ZERO(A);
#pragma omp target map(tofrom:A)
#pragma omp parallel for linear(l:2)
    for(int i = 0 ; i < 10 ; i++)
      A[i] = l;

    int fail = 0;
    for(int i = 0 ; i < 10 ; i++)
      if(A[i] != i*2) {
	printf("error at %d, val = %lf expected = %d\n", i, A[i], i*2);
	fail = 1;
      }
    if(fail)
      printf("Error\n");
    else
      printf("Succeeded\n");
  } else {
    DUMP_SUCCESS(1);
  }

  return 0;
}

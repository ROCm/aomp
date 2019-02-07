
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

  int a[N], aa[N];

  INIT();

  //
  // Test: proc_bind clause
  //
  printf("A\n");
  #undef PARALLEL_FOR_CLAUSES
  #define PARALLEL_FOR_CLAUSES proc_bind(master)
  #include "defines.h"
  for (int t = 0; t <= 224; t++) {
    int threads[1]; threads[0] = t;
    PARALLEL_FOR(
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
  printf("B\n");
  #undef PARALLEL_FOR_CLAUSES
  #define PARALLEL_FOR_CLAUSES proc_bind(close)
  #include "defines.h"
  for (int t = 0; t <= 224; t++) {
    int threads[1]; threads[0] = t;
    PARALLEL_FOR(
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
  printf("C\n");
  #undef PARALLEL_FOR_CLAUSES
  #define PARALLEL_FOR_CLAUSES proc_bind(spread)
  #include "defines.h"
  for (int t = 0; t <= 224; t++) {
    int threads[1]; threads[0] = t;
    PARALLEL_FOR(
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
  printf("D\n");
  #undef PARALLEL_FOR_CLAUSES
  #define PARALLEL_FOR_CLAUSES 
  #include "defines.h"
  for (int t = 0; t <= 224; t++) {
    int threads[1]; threads[0] = t;
    PARALLEL_FOR(
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
  // Test: private, shared clauses on omp parallel for.
  //
  printf("E\n");
  #undef PARALLEL_FOR_CLAUSES
  #define PARALLEL_FOR_CLAUSES private(p,q) shared(A,B,C,D,E)
  #include "defines.h"
  for (int t = 0; t <= 224; t++) {
    int threads[1]; threads[0] = t;
    PARALLEL_FOR(
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
  // Test: firstprivate clause on omp parallel for.
  //
  printf("F\n");
  #undef PARALLEL_FOR_CLAUSES
  #define PARALLEL_FOR_CLAUSES firstprivate(p,q)
  #include "defines.h"
  for (int t = 0; t <= 224; t++) {
    int threads[1]; threads[0] = t;
    PARALLEL_FOR(
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
  // Test: lastprivate clause on omp parallel for.
  //
  printf("G\n");
  double q0[1], q1[1], q2[1], q3[1], q4[1], q5[1], q6[1], q7[1], q8[1], q9[1];
  for (int t = 0; t <= 224; t++) {
    int threads[1]; threads[0] = t;
    TEST({
      S[0] = 0;
      for (int i = 0; i < N; i++) {
        A[i] = B[i] = 0;
      }
      _Pragma("omp parallel for simd lastprivate(q0) if(threads[0] > 1) num_threads(threads[0])")
      for (int i = 0; i < N; i++) {
        q0[0] = C[i] + D[i];
        A[i] += q0[0];
      }
      _Pragma("omp parallel for simd schedule(auto) lastprivate(q1) if(threads[0] > 1) num_threads(threads[0])")
      for (int i = 0; i < N; i++) {
        q1[0] = D[i] + E[i];
        B[i] += q1[0];
      }
      _Pragma("omp parallel for simd schedule(dynamic) lastprivate(q2) if(threads[0] > 1) num_threads(threads[0])")
      for (int i = 0; i < N; i++) {
        q2[0] = C[i] + D[i];
        A[i] += q2[0];
      }
      _Pragma("omp parallel for simd schedule(guided) lastprivate(q3) if(threads[0] > 1) num_threads(threads[0])")
      for (int i = 0; i < N; i++) {
        q3[0] = D[i] + E[i];
        B[i] += q3[0];
      }
      _Pragma("omp parallel for simd schedule(runtime) lastprivate(q4) if(threads[0] > 1) num_threads(threads[0])")
      for (int i = 0; i < N; i++) {
        q4[0] = C[i] + D[i];
        A[i] += q4[0];
      }
      _Pragma("omp parallel for simd schedule(static) lastprivate(q5) if(threads[0] > 1) num_threads(threads[0])")
      for (int i = 0; i < N; i++) {
        q5[0] = D[i] + E[i];
        B[i] += q5[0];
      }
      _Pragma("omp parallel for simd schedule(static,1) lastprivate(q6) if(threads[0] > 1) num_threads(threads[0])")
      for (int i = 0; i < N; i++) {
        q6[0] = C[i] + D[i];
        A[i] += q6[0];
      }
      _Pragma("omp parallel for simd schedule(static,9) lastprivate(q7) if(threads[0] > 1) num_threads(threads[0])")
      for (int i = 0; i < N; i++) {
        q7[0] = D[i] + E[i];
        B[i] += q7[0];
      }
      _Pragma("omp parallel for simd schedule(static,13) lastprivate(q8) if(threads[0] > 1) num_threads(threads[0])")
      for (int i = 0; i < N; i++) {
        q8[0] = C[i] + D[i];
        A[i] += q8[0];
      }
      _Pragma("omp parallel for simd schedule(static,30000) lastprivate(q9) if(threads[0] > 1) num_threads(threads[0])")
      for (int i = 0; i < N; i++) {
        q9[0] = D[i] + E[i];
        B[i] += q9[0];
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
  // Test: private clause on omp parallel for.
  //
  printf("H\n");
  #undef PARALLEL_FOR_CLAUSES
  #define PARALLEL_FOR_CLAUSES private(p)
  #include "defines.h"
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

  //
  // Test: firstprivate clause on omp parallel for.
  //

  printf("I\n");
  #undef PARALLEL_FOR_CLAUSES
  #define PARALLEL_FOR_CLAUSES firstprivate(p)
  #include "defines.h"
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

  //
  // Test: collapse clause on omp parallel for.
  //
  printf("J\n");
  #undef PARALLEL_FOR_CLAUSES
  #define PARALLEL_FOR_CLAUSES collapse(2)
  #include "defines.h"
  for (int t = 0; t <= 224; t++) {
    int threads[1]; threads[0] = t;
    PARALLEL_FOR(
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
  // Test: Ensure coalesced scheduling on GPU.
  //
  printf("K\n");
  if (omp_is_initial_device()) {
    printf ("Succeeded\n");
  } else {
    TESTD("omp target teams num_teams(1) thread_limit(33)", {
      S[0] = 0;
      for (int i = 0; i < 99; i++) {
        A[i] = 0;
      }
      _Pragma("omp parallel for simd")
      for (int i = 0; i < 99; i++) {
        A[i] += i - omp_get_thread_num();
      }
      _Pragma("omp parallel for simd schedule(auto)")
      for (int i = 0; i < 99; i++) {
        A[i] += i - omp_get_thread_num();
      } 
      _Pragma("omp parallel for simd schedule(static,1)")
      for (int i = 0; i < 99; i++) {
        A[i] += i - omp_get_thread_num();
      }
      double tmp = 0;
      for (int i = 0; i < 99; i++) {
        tmp += A[i];
      }
      S[0] = tmp;
    }, VERIFY(0, 1, S[0], 3 * (33*33 + 66*33) ));
  }

  //
  // Test: collapse clause on omp parallel for.
  //
  printf("L\n");

  int *b;

  #undef PARALLEL_FOR_CLAUSES
  #define PARALLEL_FOR_CLAUSES aligned(b: 8*sizeof(int))
  #include "defines.h"

  int threads[1];
  threads[0] = 128;

  for(int i=0; i<N; i++)
    aa[i] = a[i] = -1;

  PARALLEL_FOR(
    b = a;
  ,
  for(int k=0; k<N; k++) \
      b[k] += k;
  ,
  {
    for(int i=0; i<N; i++)
      aa[i] += i * SUMS;
  },
  VERIFY_ARRAY(0, N, aa, b));

  //
  // Test: lastprivate clause on omp parallel for.
  //
  printf("M\n");

  for(int i=0; i<N; i++)
    aa[i] = a[i] = -1;

  #undef PARALLEL_FOR_CLAUSES
  #define PARALLEL_FOR_CLAUSES lastprivate(lp)
  #include "defines.h"

  PARALLEL_FOR(
    int lp;
  ,
  for(int k=0; k<N; k++) { \
      a[k] += k; \
      lp = k; \
    } \
    a[0] += lp;
  ,
  {
    for(int i=0; i<N; i++)
      aa[i] += i * SUMS;
    aa[0] += SUMS * (N - 1);
  },
  VERIFY_ARRAY(0, N, aa, a));

  //
  // Test: lastprivate clause on omp parallel for.
  //
  printf("N\n");

  for(int i=0; i<N; i++)
    aa[i] = a[i] = -1;

  #undef PARALLEL_FOR_CLAUSES
  #define PARALLEL_FOR_CLAUSES linear(l: 2)
  #include "defines.h"

  PARALLEL_FOR(
    int l;
  ,
  for(int k=0; k<N; k++) { \
      l = 2*k; \
      a[k] += l; \
    }
  ,
  {
    for(int i=0; i<N; i++)
      aa[i] += 2 * i * SUMS;
  },
  VERIFY_ARRAY(0, N, aa, a));

  //
  // Test: lastprivate clause on omp parallel for.
  //
  printf("O\n");

  for(int i=0; i<N; i++)
    aa[i] = a[i] = -1;

  #undef PARALLEL_FOR_CLAUSES
  #define PARALLEL_FOR_CLAUSES private(p)
  #include "defines.h"

  PARALLEL_FOR(
    int p;
  ,
  for(int k=0; k<N; k++) { \
      p = k; \
      a[k] += p; \
    }
  ,
  {
    for(int i=0; i<N; i++)
      aa[i] += i * SUMS;
  },
  VERIFY_ARRAY(0, N, aa, a));

  //
  // Test: safelen clause on omp parallel for.
  //
  printf("P\n");

  for(int i=0; i<N; i++)
    aa[i] = a[i] = -1;

  #undef PARALLEL_FOR_CLAUSES
  #define PARALLEL_FOR_CLAUSES safelen(2)
  #include "defines.h"

  PARALLEL_FOR(
  ,
  for(int k=0; k<N; k++) { \
    if (k > 1) \
      a[k] = a[k-2] + 2; \
    else \
      a[k] = k; \
  }
  ,
  {
    for(int i=0; i<N; i++)
      aa[i] = i;
  },
  VERIFY_ARRAY(0, N, aa, a));

  return 0;
}

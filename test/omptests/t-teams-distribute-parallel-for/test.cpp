#include <omp.h>
#include <stdio.h>

#define MAX_N 25000

#include "../utilities/check.h"
#include "../utilities/utilities.h"

#define TRIALS (1)

#define N (1024*3)

#define INIT() INIT_LOOP(N, {C[i] = 1; D[i] = i; E[i] = -i+1;})

#define ZERO(X) ZERO_ARRAY(N, X) 

#define DUMP_SUCCESS6() { \
  if (cpuExec) { \
    DUMP_SUCCESS(3*6); \
  } \
}

void reset_input(double *a, double *a_h, double *b, double *c) {
  for(int i = 0 ; i < MAX_N ; i++) {
    a[i] = a_h[i] = i;
    b[i] = i*2;
    c[i] = i-3;
  }
}

int main(int argc, char *argv[]) {
  check_offloading();

  double A[N], B[N], C[N], D[N], E[N];
  double S[N];
  double p[2];

  int cpuExec = 0;
  #pragma omp target map(tofrom: cpuExec)
  {
    cpuExec = omp_is_initial_device();
  }
  int max_teams = 256;
  int gpu_threads = 256;
  int cpu_threads = 32;
  int max_threads = cpuExec ? cpu_threads : gpu_threads;

  INIT();

  //
  // Test: proc_bind clause
  //
  #undef TDPARALLEL_FOR_CLAUSES
  #define TDPARALLEL_FOR_CLAUSES proc_bind(master)
  #include "tdpf_defines.h"
  for (int tms = 1; tms <= max_teams; tms *= 3) {
  for (int t = 1; t <= max_threads; t+=78) {
    int threads[1]; threads[0] = t;
    int num_teams = cpuExec? 1 : tms;
    TDPARALLEL_FOR1(
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
    VERIFY(0, 1, S[0], (double) N/2*(N+1)))

    TDPARALLEL_FOR2(
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
    VERIFY(0, 1, S[0], (double) N/2*(N+1)))

    TDPARALLEL_FOR3(
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
    VERIFY(0, 1, S[0], (double) N/2*(N+1)))

    TDPARALLEL_FOR4(
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
    VERIFY(0, 1, S[0], (double) N/2*(N+1)))

    TDPARALLEL_FOR5(
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
    VERIFY(0, 1, S[0], (double) N/2*(N+1)))

    TDPARALLEL_FOR6(
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
    VERIFY(0, 1, S[0], (double) N/2*(N+1)))
  }
  DUMP_SUCCESS6()
  }

  #undef TDPARALLEL_FOR_CLAUSES
  #define TDPARALLEL_FOR_CLAUSES proc_bind(close)
  #include "tdpf_defines.h"
  for (int tms = 1; tms <= max_teams; tms *= 3) {
  for (int t = 1; t <= max_threads; t+=78) {
    int threads[1]; threads[0] = t;
    int num_teams = cpuExec? 1 : tms;
    TDPARALLEL_FOR1(
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
    VERIFY(0, 1, S[0], (double) N/2*(N+1)))

    TDPARALLEL_FOR2(
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
    VERIFY(0, 1, S[0], (double) N/2*(N+1)))

    TDPARALLEL_FOR3(
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
    VERIFY(0, 1, S[0], (double) N/2*(N+1)))

    TDPARALLEL_FOR4(
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
    VERIFY(0, 1, S[0], (double) N/2*(N+1)))

    TDPARALLEL_FOR5(
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
    VERIFY(0, 1, S[0], (double) N/2*(N+1)))

    TDPARALLEL_FOR6(
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
    VERIFY(0, 1, S[0], (double) N/2*(N+1)))
  }
  DUMP_SUCCESS6()
  }

  #undef TDPARALLEL_FOR_CLAUSES
  #define TDPARALLEL_FOR_CLAUSES proc_bind(spread)
  #include "tdpf_defines.h"
  for (int tms = 1; tms <= max_teams; tms *= 3) {
  for (int t = 1; t <= max_threads; t+=78) {
    int threads[1]; threads[0] = t;
    int num_teams = cpuExec? 1 : tms;
    TDPARALLEL_FOR1(
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
    VERIFY(0, 1, S[0], (double) N/2*(N+1)))

    TDPARALLEL_FOR2(
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
    VERIFY(0, 1, S[0], (double) N/2*(N+1)))

    TDPARALLEL_FOR3(
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
    VERIFY(0, 1, S[0], (double) N/2*(N+1)))

    TDPARALLEL_FOR4(
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
    VERIFY(0, 1, S[0], (double) N/2*(N+1)))

    TDPARALLEL_FOR5(
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
    VERIFY(0, 1, S[0], (double) N/2*(N+1)))

    TDPARALLEL_FOR6(
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
    VERIFY(0, 1, S[0], (double) N/2*(N+1)))
  }
  DUMP_SUCCESS6()
  }

  //
  // Test: private, shared clauses on omp target teams distribute parallel for.
  //
  #undef TDPARALLEL_FOR_CLAUSES
  #define TDPARALLEL_FOR_CLAUSES private(p,q) shared(A,B,C,D,E)
  #include "tdpf_defines.h"
  // FIXME: shared(a) where 'a' is an implicitly mapped scalar does not work.
  // FIXME: shared(A) private(A) does not generate correct results.
  for (int tms = 1; tms <= max_teams; tms *= 3) {
  for (int t = 1; t <= max_threads; t+=78) {
    int threads[1]; threads[0] = t;
    int num_teams = cpuExec? 1 : tms;
    TDPARALLEL_FOR1(
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
    VERIFY(0, 1, S[0], (double) 6 + N/2*(N+1)))

    TDPARALLEL_FOR2(
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
    VERIFY(0, 1, S[0], (double) 6 + N/2*(N+1)))

    TDPARALLEL_FOR3(
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
    VERIFY(0, 1, S[0], (double) 6 + N/2*(N+1)))

    TDPARALLEL_FOR4(
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
    VERIFY(0, 1, S[0], (double) 6 + N/2*(N+1)))

    TDPARALLEL_FOR5(
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
    VERIFY(0, 1, S[0], (double) 6 + N/2*(N+1)))

    TDPARALLEL_FOR6(
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
    VERIFY(0, 1, S[0], (double) 6 + N/2*(N+1)))
  }
  DUMP_SUCCESS6()
  }

  //
  // Test: firstprivate clause on omp target teams distribute parallel for.
  //
  #undef TDPARALLEL_FOR_CLAUSES
  #define TDPARALLEL_FOR_CLAUSES firstprivate(p,q)
  #include "tdpf_defines.h"
  for (int tms = 1; tms <= max_teams; tms *= 3) {
  for (int t = 1; t <= max_threads; t+=78) {
    int threads[1]; threads[0] = t;
    int num_teams = cpuExec? 1 : tms;
    TDPARALLEL_FOR1(
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
    VERIFY(0, 1, S[0], (double) N/2*(N+1)))

    TDPARALLEL_FOR2(
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
    VERIFY(0, 1, S[0], (double) N/2*(N+1)))

    TDPARALLEL_FOR3(
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
    VERIFY(0, 1, S[0], (double) N/2*(N+1)))

    TDPARALLEL_FOR4(
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
    VERIFY(0, 1, S[0], (double) N/2*(N+1)))

    TDPARALLEL_FOR5(
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
    VERIFY(0, 1, S[0], (double) N/2*(N+1)))

    TDPARALLEL_FOR6(
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
    VERIFY(0, 1, S[0], (double) N/2*(N+1)))
  }
  DUMP_SUCCESS6()
  }

#if 0
  FIXME
  //
  // Test: lastprivate clause on omp target teams distribute parallel for.
  //
  #undef TDPARALLEL_FOR_CLAUSES
  #define TDPARALLEL_FOR_CLAUSES lastprivate(q)
  #include "tdpf_defines.h"
  // FIXME: modify to t=1 and in tdpf_defines.h to use host after bug fix.
  // FIXME: variable is not private.
  for (int tms = 1; tms <= max_teams; tms *= 3) {
  for (int t = 0; t <= max_threads; t++) {
    int threads[1]; threads[0] = t;
    int num_teams = tms;
    TDPARALLEL_FOR1(
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
    VERIFY(0, 1, S[0], (double) N+1+ N/2*(N+1)))
  }

  FIXME: private of non-scalar does not work.
  //
  // Test: private clause on omp parallel for.
  //
  #undef PARALLEL_FOR_CLAUSES
  #define PARALLEL_FOR_CLAUSES private(p)
  #include "tdpf_defines.h"
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
    VERIFY(0, 1, S[0], (double) 6 + SUMS * (N/2*(N+1))))
  }

  FIXME: private of non-scalar does not work.
  //
  // Test: firstprivate clause on omp parallel for.
  //
  #undef PARALLEL_FOR_CLAUSES
  #define PARALLEL_FOR_CLAUSES firstprivate(p)
  #include "tdpf_defines.h"
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
    VERIFY(0, 1, S[0], (double) SUMS * (N/2*(N+1))))
  }
  }
#endif

  //
  // Test: collapse clause on omp target teams distribute parallel for.
  //
  #undef TDPARALLEL_FOR_CLAUSES
  #define TDPARALLEL_FOR_CLAUSES collapse(2)
  #include "tdpf_defines.h"
  for (int tms = 1; tms <= max_teams; tms *= 3) {
  for (int t = 1; t <= max_threads; t+=78) {
    int threads[1]; threads[0] = t;
    int num_teams = cpuExec? 1 : tms;
    TDPARALLEL_FOR1(
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
    VERIFY(0, 1, S[0], (double) (N/2*(N+1))))

    TDPARALLEL_FOR2(
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
    VERIFY(0, 1, S[0], (double) (N/2*(N+1))))

    TDPARALLEL_FOR3(
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
    VERIFY(0, 1, S[0], (double) (N/2*(N+1))))

    TDPARALLEL_FOR4(
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
    VERIFY(0, 1, S[0], (double) (N/2*(N+1))))

    TDPARALLEL_FOR5(
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
    VERIFY(0, 1, S[0], (double) (N/2*(N+1))))

    TDPARALLEL_FOR6(
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
    VERIFY(0, 1, S[0], (double) (N/2*(N+1))))
  }
  DUMP_SUCCESS6()
  }

  double * a = (double *) malloc(MAX_N * sizeof(double));
  double * a_h = (double *) malloc(MAX_N * sizeof(double));
  double * b = (double *) malloc(MAX_N * sizeof(double));
  double * c = (double *) malloc(MAX_N * sizeof(double));

#pragma omp target enter data map(to:a[:MAX_N],b[:MAX_N],c[:MAX_N])

  // 1. no schedule clauses
  printf("no schedule clauses\n");
  for (int n = 32 ; n < MAX_N ; n+=5000) {

    reset_input(a, a_h, b, c);

#pragma omp target update to(a[:n],b[:n],c[:n])
    int t = 0;
    for (int tms = 1 ; tms <= 256 ; tms *= 2) { // 8 times
      for (int ths = 32 ; ths <= 1024 ; ths *= 2) { // 6 times
	t++;
#pragma omp target
#pragma omp teams distribute parallel for num_teams(tms) thread_limit(ths)
	  for (int i = 0; i < n; ++i) {
	    a[i] += b[i] + c[i];
	  }
      } // loop over 'ths'
    } // loop over 'tms'

    // check results for each 'n'
    for (int times = 0 ; times < t ; times++)
      for (int i = 0; i < n; ++i)
	a_h[i] += b[i] + c[i];

#pragma omp target update from(a[:n])

    for (int i = 0; i < n; ++i) {
      if (a_h[i] != a[i]) {
	printf("Error at n = %d, i = %d: host = %lf, device = %lf\n", n, i, a_h[i], a[i]);
	return 1;
      }
    }
  } // loop over 'n'
  printf("Succeeded\n");

  // 2. schedule static no chunk
  printf("schedule static no chunk\n");
  for (int n = 32 ; n < MAX_N ; n+=5000) {

    reset_input(a, a_h, b, c);

#pragma omp target update to(a[:n],b[:n],c[:n])
    int t = 0;
    for (int tms = 1 ; tms <= 256 ; tms *= 2) { // 8 times
      for (int ths = 32 ; ths <= 1024 ; ths *= 2) { // 6 times
	t++;
#pragma omp target
#pragma omp teams distribute parallel for schedule(static) num_teams(tms) thread_limit(ths)
	  for (int i = 0; i < n; ++i) {
	    a[i] += b[i] + c[i];
	  }
      } // loop over 'ths'
    } // loop over 'tms'

    // check results for each 'n'
    for (int times = 0 ; times < t ; times++)
      for (int i = 0; i < n; ++i)
	a_h[i] += b[i] + c[i];

#pragma omp target update from(a[:n])

    for (int i = 0; i < n; ++i) {
      if (a_h[i] != a[i]) {
	printf("Error at n = %d, i = %d: host = %lf, device = %lf\n", n, i, a_h[i], a[i]);
	return 1;
      }
    }
  } // loop over 'n'
  printf("Succeeded\n");

  // 3. schedule static chunk
  printf("schedule static chunk\n");
  for (int n = 32 ; n < MAX_N ; n+=5000) {
    int t = 0;
    reset_input(a, a_h, b, c);

#pragma omp target update to(a[:n],b[:n],c[:n])

    for (int tms = 1 ; tms <= 256 ; tms *= 2) { // 8 times
      for (int ths = 32 ; ths <= 1024 ; ths *= 2) { // 6 times
	for(int sch = 1 ; sch <= n ; sch *= 3000) {
	  t++;
#pragma omp target
#pragma omp teams distribute parallel for schedule(static,sch) num_teams(tms) thread_limit(ths)
	    for (int i = 0; i < n; ++i) {
	      a[i] += b[i] + c[i];
	    }
	} // loop 'sch'
      } // loop 'ths'
    } // loop 'tms'

    // check results for each 'n'
    for (int times = 0 ; times < t ; times++)
      for (int i = 0; i < n; ++i)
	a_h[i] += b[i] + c[i];

#pragma omp target update from(a[:n])

    for (int i = 0; i < n; ++i) {
      if (a_h[i] != a[i]) {
	printf("Error at n = %d, i = %d: host = %lf, device = %lf\n", n, i, a_h[i], a[i]);
	return 1;
      }
    }
  } // loop 'n'
  printf("Succeeded\n");

  // 4. schedule dynamic no chunk (debugging)
  printf("schedule dynamic no chunk\n");
  for (int n = 32 ; n < MAX_N ; n+=5000) {
    int t = 0;
    reset_input(a, a_h, b, c);

#pragma omp target update to(a[:n],b[:n],c[:n])

    for (int tms = 1 ; tms <= 256 ; tms *= 2) { // 8 times
      for (int ths = 32 ; ths <= 1024 ; ths *= 2) { // 6 times
	  t++;
#pragma omp target
#pragma omp teams distribute parallel for schedule(dynamic) num_teams(tms) thread_limit(ths)
	    for (int i = 0; i < n; ++i) {
	      a[i] += b[i] + c[i];
	    }
      } // loop 'ths'
    } // loop 'tms'


    // check results for each 'n'
    for (int times = 0 ; times < t ; times++)
      for (int i = 0; i < n; ++i)
	a_h[i] += b[i] + c[i];

#pragma omp target update from(a[:n])

    for (int i = 0; i < n; ++i) {
      if (a_h[i] != a[i]) {
	printf("Error at n = %d, i = %d: host = %lf, device = %lf\n", n, i, a_h[i], a[i]);
	return 1;
      }
    }
  } // loop 'n'
  printf("Succeeded\n");


  // 5. schedule dynamic chunk (debugging)
  printf("schedule dynamic chunk\n");
  for (int n = 32 ; n < MAX_N ; n+=5000) {
    int t = 0;
    reset_input(a, a_h, b, c);

#pragma omp target update to(a[:n],b[:n],c[:n])

    for (int tms = 1 ; tms <= 256 ; tms *= 2) { // 8 times
      for (int ths = 32 ; ths <= 1024 ; ths *= 2) { // 6 times
	for(int sch = 1 ; sch <= n ; sch *= 1200) {
	  t++;
#pragma omp target
#pragma omp teams distribute parallel for schedule(dynamic, sch) num_teams(tms) thread_limit(ths)
	    for (int i = 0; i < n; ++i) {
	      a[i] += b[i] + c[i];
	    }
	} // loop 'sch'
      } // loop 'ths'
    } // loop 'tms'


    // check results for each 'n'
    for (int times = 0 ; times < t ; times++)
      for (int i = 0; i < n; ++i)
	a_h[i] += b[i] + c[i];

#pragma omp target update from(a[:n])

    for (int i = 0; i < n; ++i) {
      if (a_h[i] != a[i]) {
	printf("Error at n = %d, i = %d: host = %lf, device = %lf\n", n, i, a_h[i], a[i]);
	return 1;
      }
    }
  } // loop 'n'
  printf("Succeeded\n");

  // 6. dist_schedule static no chunk
  printf("dist_schedule static no chunk\n");
  for (int n = 32 ; n < MAX_N ; n+=5000) {

    reset_input(a, a_h, b, c);

#pragma omp target update to(a[:n],b[:n],c[:n])
    int t = 0;
    for (int tms = 1 ; tms <= 256 ; tms *= 2) { // 8 times
      for (int ths = 32 ; ths <= 1024 ; ths *= 2) { // 6 times
	t++;
#pragma omp target
#pragma omp teams distribute parallel for dist_schedule(static) num_teams(tms) thread_limit(ths)
	  for (int i = 0; i < n; ++i) {
	    a[i] += b[i] + c[i];
	  }
      }
    }

    // check results for each 'n'
    for (int times = 0 ; times < t ; times++)
      for (int i = 0; i < n; ++i)
	a_h[i] += b[i] + c[i];

#pragma omp target update from(a[:n])

    for (int i = 0; i < n; ++i) {
      if (a_h[i] != a[i]) {
	printf("Error at n = %d, i = %d: host = %lf, device = %lf\n", n, i, a_h[i], a[i]);
	return 1;
      }
    }
  } // loop over 'n'
  printf("Succeeded\n");

  // 7. dist_schedule static chunk
  printf("dist_schedule static chunk\n");
  for (int n = 32 ; n < MAX_N ; n+=5000) {
    int t = 0;
    reset_input(a, a_h, b, c);

#pragma omp target update to(a[:n],b[:n],c[:n])

    for (int tms = 1 ; tms <= 256 ; tms *= 2) { // 8 times
      for (int ths = 32 ; ths <= 1024 ; ths *= 2) { // 6 times
	for(int sch = 128 ; sch <= n ; sch *= 10000) {
	  t++;
#pragma omp target
#pragma omp teams distribute parallel for dist_schedule(static,sch) num_teams(tms) thread_limit(ths)
	    for (int i = 0; i < n; ++i) {
	      a[i] += b[i] + c[i];
	    }
	} // loop 'sch'
      } // loop 'ths'
    } // loop 'tms'

    // check results for each 'n'
    for (int times = 0 ; times < t ; times++)
      for (int i = 0; i < n; ++i)
	a_h[i] += b[i] + c[i];

#pragma omp target update from(a[:n])

    for (int i = 0; i < n; ++i) {
      if (a_h[i] != a[i]) {
	printf("Error at n = %d, i = %d: host = %lf, device = %lf\n", n, i, a_h[i], a[i]);
	return 1;
      }
    }
  } // loop 'n'
  printf("Succeeded\n");

  // 8. dist_schedule static no chunk, schedule static no chunk
  printf("dist_schedule static no chunk, schedule static no chunk\n");
  for (int n = 32 ; n < MAX_N ; n+=5000) {

    reset_input(a, a_h, b, c);

#pragma omp target update to(a[:n],b[:n],c[:n])
    int t = 0;
    for (int tms = 1 ; tms <= 256 ; tms *= 2) { // 8 times
      for (int ths = 32 ; ths <= 1024 ; ths *= 2) { // 6 times
	t++;
#pragma omp target
#pragma omp teams distribute parallel for dist_schedule(static) schedule(static) num_teams(tms) thread_limit(ths)
	  for (int i = 0; i < n; ++i) {
	    a[i] += b[i] + c[i];
	  }
      }
    }

    // check results for each 'n'
    for (int times = 0 ; times < t ; times++)
      for (int i = 0; i < n; ++i)
	a_h[i] += b[i] + c[i];

#pragma omp target update from(a[:n])

    for (int i = 0; i < n; ++i) {
      if (a_h[i] != a[i]) {
	printf("Error at n = %d, i = %d: host = %lf, device = %lf\n", n, i, a_h[i], a[i]);
	return 1;
      }
    }
  } // loop over 'n'
  printf("Succeeded\n");

  // 9. dist_schedule static no chunk, schedule static chunk
  printf("dist_schedule static no chunk, schedule static chunk\n");
  for (int n = 32 ; n < MAX_N ; n+=5000) {
    int t = 0;
    reset_input(a, a_h, b, c);

#pragma omp target update to(a[:n],b[:n],c[:n])

    for (int tms = 1 ; tms <= 256 ; tms *= 2) { // 8 times
      for (int ths = 32 ; ths <= 1024 ; ths *= 2) { // 6 times
	for(int sch = 1 ; sch <= n ; sch *= 1000) { // speed up very slow tests
	  t++;
#pragma omp target
#pragma omp teams distribute parallel for dist_schedule(static) schedule(static,sch) num_teams(tms) thread_limit(ths)
	    for (int i = 0; i < n; ++i) {
	      a[i] += b[i] + c[i];
	    }
	} // loop 'sch'
      } // loop 'ths'
    } // loop 'tms'

    // check results for each 'n'
    for (int times = 0 ; times < t ; times++)
      for (int i = 0; i < n; ++i)
	a_h[i] += b[i] + c[i];

#pragma omp target update from(a[:n])

    for (int i = 0; i < n; ++i) {
      if (a_h[i] != a[i]) {
	printf("Error at n = %d, i = %d: host = %lf, device = %lf\n", n, i, a_h[i], a[i]);
	return 1;
      }
    }
  } // loop 'n'
  printf("Succeeded\n");

  // 10. dist_schedule static chunk, schedule static no chunk
  printf("dist_schedule static chunk, schedule static no chunk\n");
  for (int n = 32 ; n < MAX_N ; n+=5000) {
    int t = 0;
    reset_input(a, a_h, b, c);

#pragma omp target update to(a[:n],b[:n],c[:n])

    for (int tms = 1 ; tms <= 256 ; tms *= 2) { // 8 times
      for (int ths = 32 ; ths <= 1024 ; ths *= 2) { // 6 times
	for(int sch = 128 ; sch <= n ; sch *= 1200) {
	  t++;
#pragma omp target
#pragma omp teams distribute parallel for dist_schedule(static,sch) schedule(static) num_teams(tms) thread_limit(ths)
	    for (int i = 0; i < n; ++i) {
	      a[i] += b[i] + c[i];
	    }
	} // loop 'sch'
      } // loop 'ths'
    } // loop 'tms'

    // check results for each 'n'
    for (int times = 0 ; times < t ; times++)
      for (int i = 0; i < n; ++i)
	a_h[i] += b[i] + c[i];

#pragma omp target update from(a[:n])

    for (int i = 0; i < n; ++i) {
      if (a_h[i] != a[i]) {
	printf("Error at n = %d, i = %d: host = %lf, device = %lf\n", n, i, a_h[i], a[i]);
	return 1;
      }
    }
  } // loop 'n'
  printf("Succeeded\n");

  // 11. dist_schedule static chunk, schedule static chunk
  printf("dist_schedule static chunk, schedule static chunk\n");
  for (int n = 32 ; n < MAX_N ; n+=5000) {
    int t = 0;
    reset_input(a, a_h, b, c);

#pragma omp target update to(a[:n],b[:n],c[:n])

    for (int tms = 1 ; tms <= 256 ; tms *= 2) { // 8 times
      for (int ths = 32 ; ths <= 1024 ; ths *= 2) { // 6 times
	for(int dssch = 128 ; dssch <= n ; dssch *= 1200) {
	  for(int sch = 100 ; sch <= n ; sch *= 3000) {
	    t++;
#pragma omp target
#pragma omp teams distribute parallel for dist_schedule(static,dssch) schedule(static,sch) num_teams(tms) thread_limit(ths)
	      for (int i = 0; i < n; ++i) {
		a[i] += b[i] + c[i];
	      }
	  } // loop 'sch'
	} // loop 'dssch'
      } // loop 'ths'
    } // loop 'tms'

    // check results for each 'n'
    for (int times = 0 ; times < t ; times++)
      for (int i = 0; i < n; ++i)
	a_h[i] += b[i] + c[i];

#pragma omp target update from(a[:n])

    for (int i = 0; i < n; ++i) {
      if (a_h[i] != a[i]) {
	printf("Error at n = %d, i = %d: host = %lf, device = %lf\n", n, i, a_h[i], a[i]);
	return 1;
      }
    }
  } // loop 'n'
  printf("Succeeded\n");

  // 12. dist_schedule static chunk, schedule dynamic no chunk
  printf("dist_schedule static chunk, schedule dynamic no chunk\n");
  for (int n = 32 ; n < MAX_N ; n+=5000) {
    int t = 0;
    reset_input(a, a_h, b, c);

#pragma omp target update to(a[:n],b[:n],c[:n])

    for (int tms = 1 ; tms <= 256 ; tms *= 2) { // 8 times
      for (int ths = 32 ; ths <= 1024 ; ths *= 2) { // 6 times
	for(int sch = 128 ; sch <= n ; sch *= 3000) {
	  t++;
#pragma omp target
#pragma omp teams distribute parallel for dist_schedule(static,sch) schedule(dynamic) num_teams(tms) thread_limit(ths)
	    for (int i = 0; i < n; ++i) {
	      a[i] += b[i] + c[i];
	    }
	} // loop 'sch'
      } // loop 'ths'
    } // loop 'tms'

    // check results for each 'n'
    for (int times = 0 ; times < t ; times++)
      for (int i = 0; i < n; ++i)
	a_h[i] += b[i] + c[i];

#pragma omp target update from(a[:n])

    for (int i = 0; i < n; ++i) {
      if (a_h[i] != a[i]) {
	printf("Error at n = %d, i = %d: host = %lf, device = %lf\n", n, i, a_h[i], a[i]);
	return 1;
      }
    }
  } // loop 'n'
  printf("Succeeded\n");

  // 13. dist_schedule static chunk, schedule dynamic chunk
  printf("dist_schedule static chunk, schedule dynamic chunk\n");
  for (int n = 32 ; n < MAX_N ; n+=5000) {
    int t = 0;
    reset_input(a, a_h, b, c);

#pragma omp target update to(a[:n],b[:n],c[:n])

    for (int tms = 1 ; tms <= 256 ; tms *= 2) { // 8 times
      for (int ths = 32 ; ths <= 1024 ; ths *= 2) { // 6 times
	for(int dssch = 128 ; dssch <= n ; dssch *= 3000) {
	  for(int sch = 1000 ; sch <= n ; sch *= 3000) {
	    t++;
#pragma omp target
#pragma omp teams distribute parallel for dist_schedule(static,dssch) schedule(dynamic,sch) num_teams(tms) thread_limit(ths)
	      for (int i = 0; i < n; ++i) {
		a[i] += b[i] + c[i];
	      }
	  } // loop 'sch'
	} // loop 'dssch'
      } // loop 'ths'
    } // loop 'tms'

    // check results for each 'n'
    for (int times = 0 ; times < t ; times++)
      for (int i = 0; i < n; ++i)
	a_h[i] += b[i] + c[i];

#pragma omp target update from(a[:n])

    for (int i = 0; i < n; ++i) {
      if (a_h[i] != a[i]) {
	printf("Error at n = %d, i = %d: host = %lf, device = %lf\n", n, i, a_h[i], a[i]);
	return 1;
      }
    }
  } // loop 'n'
  printf("Succeeded\n");

  // 14. dist_schedule static chunk, schedule guided no chunk
  printf("dist_schedule static chunk, schedule guided no chunk\n");
  for (int n = 32 ; n < MAX_N ; n+=5000) {
    int t = 0;
    reset_input(a, a_h, b, c);

#pragma omp target update to(a[:n],b[:n],c[:n])

    for (int tms = 1 ; tms <= 256 ; tms *= 2) { // 8 times
      for (int ths = 32 ; ths <= 1024 ; ths *= 2) { // 6 times
	for(int sch = 1000 ; sch <= n ; sch *= 3000) {
	  t++;
#pragma omp target
#pragma omp teams distribute parallel for dist_schedule(static,sch) schedule(guided) num_teams(tms) thread_limit(ths)
	    for (int i = 0; i < n; ++i) {
	      a[i] += b[i] + c[i];
	    }
	} // loop 'sch'
      } // loop 'ths'
    } // loop 'tms'

    // check results for each 'n'
    for (int times = 0 ; times < t ; times++)
      for (int i = 0; i < n; ++i)
	a_h[i] += b[i] + c[i];

#pragma omp target update from(a[:n])

    for (int i = 0; i < n; ++i) {
      if (a_h[i] != a[i]) {
	printf("Error at n = %d, i = %d: host = %lf, device = %lf\n", n, i, a_h[i], a[i]);
	return 1;
      }
    }
  } // loop 'n'
  printf("Succeeded\n");

  // 15. dist_schedule static chunk, schedule guided chunk
  printf("dist_schedule static chunk, schedule guided chunk\n");
  for (int n = 32 ; n < MAX_N ; n+=5000) {
    int t = 0;
    reset_input(a, a_h, b, c);

#pragma omp target update to(a[:n],b[:n],c[:n])

    for (int tms = 1 ; tms <= 256 ; tms *= 2) { // 8 times
      for (int ths = 32 ; ths <= 1024 ; ths *= 2) { // 6 times
	for(int dssch = 1000 ; dssch <= n ; dssch *= 3000) {
	  for(int sch = 1000 ; sch <= n ; sch *= 3000) {
	    t++;
#pragma omp target
#pragma omp teams distribute parallel for dist_schedule(static,dssch) schedule(guided,sch) num_teams(tms) thread_limit(ths)
	      for (int i = 0; i < n; ++i) {
		a[i] += b[i] + c[i];
	      }
	  } // loop 'sch'
	} // loop 'dssch'
      } // loop 'ths'
    } // loop 'tms'

    // check results for each 'n'
    for (int times = 0 ; times < t ; times++)
      for (int i = 0; i < n; ++i)
	a_h[i] += b[i] + c[i];

#pragma omp target update from(a[:n])

    for (int i = 0; i < n; ++i) {
      if (a_h[i] != a[i]) {
	printf("Error at n = %d, i = %d: host = %lf, device = %lf\n", n, i, a_h[i], a[i]);
	return 1;
      }
    }
  } // loop 'n'
  printf("Succeeded\n");


  // 16. dist_schedule static chunk, schedule auto
  printf("dist_schedule static chunk, schedule auto\n");
  for (int n = 32 ; n < MAX_N ; n+=5000) {
    int t = 0;
    reset_input(a, a_h, b, c);

#pragma omp target update to(a[:n],b[:n],c[:n])

    for (int tms = 1 ; tms <= 256 ; tms *= 2) { // 8 times
      for (int ths = 32 ; ths <= 1024 ; ths *= 2) { // 6 times
	for(int sch = 1000 ; sch <= n ; sch *= 3000) {
	  t++;
#pragma omp target
#pragma omp teams distribute parallel for dist_schedule(static,sch) schedule(auto) num_teams(tms) thread_limit(ths)
	    for (int i = 0; i < n; ++i) {
	      a[i] += b[i] + c[i];
	    }
	} // loop 'sch'
      } // loop 'ths'
    } // loop 'tms'

    // check results for each 'n'
    for (int times = 0 ; times < t ; times++)
      for (int i = 0; i < n; ++i)
	a_h[i] += b[i] + c[i];

#pragma omp target update from(a[:n])

    for (int i = 0; i < n; ++i) {
      if (a_h[i] != a[i]) {
	printf("Error at n = %d, i = %d: host = %lf, device = %lf\n", n, i, a_h[i], a[i]);
	return 1;
      }
    }
  } // loop 'n'
  printf("Succeeded\n");

    // 17. dist_schedule static chunk, schedule runtime
  printf("dist_schedule static chunk, schedule runtime\n");
  for (int n = 32 ; n < MAX_N ; n+=5000) {
    int t = 0;
    reset_input(a, a_h, b, c);

#pragma omp target update to(a[:n],b[:n],c[:n])

    for (int tms = 1 ; tms <= 256 ; tms *= 2) { // 8 times
      for (int ths = 32 ; ths <= 1024 ; ths *= 2) { // 6 times
	for(int sch = 1000 ; sch <= n ; sch *= 3000) {
	  t++;
#pragma omp target
#pragma omp teams distribute parallel for dist_schedule(static,sch) schedule(runtime) num_teams(tms) thread_limit(ths)
	    for (int i = 0; i < n; ++i) {
	      a[i] += b[i] + c[i];
	    }
	} // loop 'sch'
      } // loop 'ths'
    } // loop 'tms'

    // check results for each 'n'
    for (int times = 0 ; times < t ; times++)
      for (int i = 0; i < n; ++i)
	a_h[i] += b[i] + c[i];

#pragma omp target update from(a[:n])

    for (int i = 0; i < n; ++i) {
      if (a_h[i] != a[i]) {
	printf("Error at n = %d, i = %d: host = %lf, device = %lf\n", n, i, a_h[i], a[i]);
	return 1;
      }
    }
  } // loop 'n'
  printf("Succeeded\n");

#pragma omp target exit data map(release:a[:MAX_N],b[:MAX_N],c[:MAX_N])

  return 0;
}

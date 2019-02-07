
#include <stdio.h>
#include <omp.h>

#include "../utilities/check.h"
#include "../utilities/utilities.h"

#define TRIALS (1)

#define N (992)

#define INIT() INIT_LOOP(N, {C[i] = 1; D[i] = i; E[i] = -i;})

#define ZERO(X) ZERO_ARRAY(N, X) 

int main(void) {
  check_offloading();

  double A[N], B[N], C[N], D[N], E[N];

  INIT();

  //
  // Test: Single.
  //
  for (int t = 0; t <= 224; t++) {
    int threads[1]; threads[0] = t;
    TEST({
    for (int i = 0; i < N; i++) {
      A[i] = 1;
      B[i] = 0;
    }
    _Pragma("omp parallel if(threads[0] > 1) num_threads(threads[0])")
      {
        _Pragma("omp for nowait schedule(static,1)")
        for (int i = 0; i < N; i++) {
          B[i] = D[i] - E[i];
        }
        _Pragma("omp single")
        {
          for (int i = 0; i < N; i++) {
            A[i] += C[i] + D[i];
          }
        }
        _Pragma("omp for schedule(static,1)")
        for (int i = 0; i < N; i++) {
          B[i] += A[i];
        }
      }
    }, VERIFY(0, N, B[i], 3*i+2));
  }

  //
  // Test: Single - private, nowait.
  //
  for (int t = 0; t <= 224; t++) {
    int threads[1]; threads[0] = t;
    TEST({
    for (int i = 0; i < N; i++) {
      A[i] = 1;
      B[i] = 0;
    }
    _Pragma("omp parallel if(threads[0] > 1) num_threads(threads[0])")
      {
        _Pragma("omp for nowait schedule(static,1)")
        for (int i = 0; i < N; i++) {
          B[i] = D[i] - E[i];
        }
        _Pragma("omp single nowait private(A)")
        {
          for (int i = 0; i < N; i++) {
            A[i] += C[i] + D[i];
          }
        }
        _Pragma("omp for schedule(static,1)")
        for (int i = 0; i < N; i++) {
          B[i] += A[i];
        }
      }
    }, VERIFY(0, N, B[i], 2*i+1));
  }

  //
  // Test: Single - firstprivate, should not have nowait.
  //
  for (int t = 0; t <= 224; t++) {
    int threads[1]; threads[0] = t;
    TEST({
    for (int i = 0; i < N; i++) {
      A[i] = 1;
      B[i] = 0;
    }
    _Pragma("omp parallel if(threads[0] > 1) num_threads(threads[0])")
      {
        _Pragma("omp for schedule(static,1)")
        for (int i = 0; i < N; i++) {
          B[i] = D[i] - E[i];
        }
        _Pragma("omp single firstprivate(A)")
        {
          for (int i = 0; i < N; i++) {
            A[i] += C[i] + D[i];
            B[i] += A[i];
          }
        }
        _Pragma("omp for schedule(static,1)")
        for (int i = 0; i < N; i++) {
          B[i] += A[i];
        }
      }
    }, VERIFY(0, N, B[i], 3*i+3));
  }

#if 0
  //
  // Test: Single - copyprivate, should not have nowait.
  //
  for (int t = 0; t <= 224; t++) {
    int threads[1]; threads[0] = t;
    TEST({
    for (int i = 0; i < N; i++) {
      A[i] = 1;
      B[i] = 0;
    }
    _Pragma("omp parallel private(A) if(threads[0] > 1) num_threads(threads[0])")
      {
        _Pragma("omp for schedule(static,1)")
        for (int i = 0; i < N; i++) {
          B[i] = D[i] - E[i];
          A[i] = i;
        }
        _Pragma("omp single copyprivate(A)")
        {
          for (int i = 0; i < N; i++) {
            A[i] += C[i] + D[i];
            B[i] += A[i];
          }
        }
      }
      for (int i = 0; i < N; i++) {
        B[i] += A[i];
      }
    }, VERIFY(0, N, B[i], 4*i+3));
  }
#endif

  return 0;
}

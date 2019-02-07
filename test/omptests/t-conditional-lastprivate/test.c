
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

  int A[N], B[N], C[N], D[N], E[N];
  int S[N];

  INIT();

  long cpuExec = 0;
  #pragma omp target map(tofrom: cpuExec)
  {
    cpuExec = omp_is_initial_device();
  }
  int max_threads = 224;

  //
  // Test: lastprivate clause on omp for.
  //
  for (int t = 0; t <= max_threads; t++) {
    int threads = t;
    TEST({
      S[0] = 0;
      double q0; double q1; double q2; double q3; double q4;
      double q5; double q6; double q7; double q8; double q9;
      q0 = q1 = q2 = q3 = q4 = q5 = q6 = q7 = q8 = q9 = 0;
    _Pragma("omp parallel if(threads > 1) num_threads(threads)")
      {
        _Pragma("omp for lastprivate(conditional: q0)")
        for (int i = 0; i < N; i++) {
          if (D[i] % 10 == 0)
            q0 = C[i] + D[i];
          A[i] += q0;
        }
        _Pragma("omp for schedule(auto) lastprivate(conditional: q1)")
        for (int i = 0; i < N; i++) {
          if (D[i] % 10 == 0)
            q1 = C[i] + D[i];
          B[i] += q1;
        }
        _Pragma("omp for schedule(dynamic) lastprivate(conditional: q2)")
        for (int i = 0; i < N; i++) {
          if (D[i] % 10 == 0)
            q2 = C[i] + D[i];
          A[i] += q2;
        }
        _Pragma("omp for schedule(guided) lastprivate(conditional: q3)")
        for (int i = 0; i < N; i++) {
          if (D[i] % 10 == 0)
            q3 = C[i] + D[i];
          B[i] += q3;
        }
        _Pragma("omp for schedule(runtime) lastprivate(conditional: q4)")
        for (int i = 0; i < N; i++) {
          if (D[i] % 10 == 0)
            q4 = C[i] + D[i];
          A[i] += q4;
        }
        _Pragma("omp for schedule(static) lastprivate(conditional: q5)")
        for (int i = 0; i < N; i++) {
          if (D[i] % 10 == 0)
            q5 = C[i] + D[i];
          B[i] += q5;
        }
        _Pragma("omp for schedule(static,1) lastprivate(conditional: q6)")
        for (int i = 0; i < N; i++) {
          if (D[i] % 10 == 0)
            q6 = C[i] + D[i];
          A[i] += q6;
        }
        _Pragma("omp for schedule(static,9) lastprivate(conditional: q7)")
        for (int i = 0; i < N; i++) {
          if (D[i] % 10 == 0)
            q7 = C[i] + D[i];
          B[i] += q7;
        }
        _Pragma("omp for schedule(static,13) lastprivate(conditional: q8)")
        for (int i = 0; i < N; i++) {
          if (D[i] % 10 == 0)
            q8 = C[i] + D[i];
          A[i] += q8;
        }
        _Pragma("omp for schedule(static,30000) lastprivate(conditional: q9)")
        for (int i = 0; i < N; i++) {
          if (D[i] % 10 == 0)
            q9 = C[i] + D[i];
          B[i] += q9;
        }
      }
      double tmp = q0 + q1 + q2 + q3 + q4 + \
                   q5 + q6 + q7 + q8 + q9;
      S[0] = tmp;
    }, VERIFY(0, 1, S[0], 30710 ));
  }

  return 0;
}

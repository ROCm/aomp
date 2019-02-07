#include <stdio.h>
#include <omp.h>

#include "../utilities/check.h"
#include "../utilities/utilities.h"

// enable tests
#define CHECK              1
#define FULL               1
#define FULL_ZERO          0  /* use zero ptrs, not legal (yet) */
#define FULL_S             0  /* need struct support */
#define OFFSET             1
#define OFFSET_S           0  /* need struct support */

#define N (992)

#define INIT() INIT_LOOP(N, {A[i] = 0; C[i] = 1; D[i] = i; E[i] = -i; })

int main(void){
#if CHECK
    check_offloading();
#endif

  int fail;
  double A[N], B[N], C[N], D[N], E[N];
  double *pA, *pB, *pC, *pD, *pE;

  // map ptrs
  pA = &A[0];
  pB = &B[0];
  pC = &C[0];
  pD = &D[0];
  pE = &E[0];

#if FULL
  INIT();
  #pragma omp target data map(from: pA[0:N]) map(to: pC[0:N], pD[0:N]) device(1)
  {
    #pragma omp target device(1)
    {
      #pragma omp parallel for schedule(static,1)
      for(int i = 0; i < N; i++)
        pA[i] = pC[i] + pD[i] + 1;
    }

    #pragma omp target update from(pA[0:N]) device(1)

    // CHECK: Succeeded in "update from"
    fail = 0;
    VERIFY(0, N, A[i], (double)(i+2));
    if (fail) {
      printf ("Test update from: Failed\n");
    } else {
      printf ("Test update from: Succeeded\n");
    }

    // Now modify host arrays C and D
    for(int i = 0; i < N; i++){
      C[i] = 2;
      D[i] = i + 1;
    }

    #pragma omp target update to(pC[0:N], pD[0:N]) device(1)

    #pragma omp target device(1)
    {
      #pragma omp parallel for schedule(static,1)
      for(int i = 0; i < N; i++)
        pA[i] = pC[i] + pD[i] + 1;
    }

    #pragma omp target update from(pA[0:N]) device(1)

    // CHECK: Succeeded in "update to"
    fail = 0;
    VERIFY(0, N, A[i], (double)(i+4));
    if (fail) {
      printf ("Test update to: Failed\n");
    } else {
      printf ("Test update to: Succeeded\n");
    }
  }
#endif

#if FULL_ZERO
  INIT();
  #pragma omp target data map(from: pA[0:N]) map(to: pC[0:N], pD[0:N]) device(1)
  {
    #pragma omp target device(1)
    {
      #pragma omp parallel for schedule(static,1)
      for(int i = 0; i < N; i++)
        pA[i] = pC[i] + pD[i] + 1;
    }

    #pragma omp target update from(pA[0:0]) device(1)

    // CHECK: Succeeded in "update from"
    fail = 0;
    VERIFY(0, N, A[i], (double)(i+2));
    if (fail) {
      printf ("Test update from with zero-length ptrs: Failed\n");
    } else {
      printf ("Test update from with zero-length ptrs: Succeeded\n");
    }

    // Now modify host arrays C and D
    for(int i = 0; i < N; i++){
      C[i] = 2;
      D[i] = i + 1;
    }

    #pragma omp target update to(pC[0:0], pD[0:0]) device(1)

    #pragma omp target device(1)
    {
      #pragma omp parallel for schedule(static,1)
      for(int i = 0; i < N; i++)
        pA[i] = pC[i] + pD[i] + 1;
    }

    #pragma omp target update from(pA[0:0]) device(1)

    // CHECK: Succeeded in "update to"
    fail = 0;
    VERIFY(0, N, A[i], (double)(i+4));
    if (fail) {
      printf ("Test update to with zero-length ptrs: Failed\n");
    } else {
      printf ("Test update to with zero-length ptrs: Succeeded\n");
    }
  }
#endif

#if OFFSET
  pA = pA - 100;
  pC = pC - 200;
  pD = pD - 300;
  INIT();
  #pragma omp target data map(from: pA[100:N]) map(to: pC[200:N], pD[300:N])\
                          device(1)
  {
    #pragma omp target map(from: pA[100:N]) map(to: pC[200:N], pD[300:N])\
                       device(1)
    {
      #pragma omp parallel for schedule(static,1)
      for(int i = 0; i < N; i++)
        pA[i+100] = pC[i+200] + pD[i+300] + 1;
    }

    #pragma omp target update from(pA[100:N]) device(1)

    // CHECK: Succeeded in "update from"
    fail = 0;
    VERIFY(0, N, A[i], (double)(i+2));
    if (fail) {
      printf ("Test update from with offsets: Failed\n");
    } else {
      printf ("Test update from with offsets: Succeeded\n");
    }

    // Now modify host arrays C and D
    for(int i = 0; i < N; i++){
      C[i] = 2;
      D[i] = i + 1;
    }

    #pragma omp target update to(pC[200:N], pD[300:N]) device(1)

    #pragma omp target map(from: pA[100:N]) map(to: pC[200:N], pD[300:N])\
                       device(1)
    {
      #pragma omp parallel for schedule(static,1)
      for(int i = 0; i < N; i++)
        pA[i+100] = pC[i+200] + pD[i+300] + 1;
    }

    #pragma omp target update from(pA[100:N]) device(1)

    // CHECK: Succeeded in "update to"
    fail = 0;
    VERIFY(0, N, A[i], (double)(i+4));
    if (fail) {
      printf ("Test update to with offsets: Failed\n");
    } else {
      printf ("Test update to with offsets: Succeeded\n");
    }
  }
#endif

  return 0;
}

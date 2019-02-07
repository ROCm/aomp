
#include <stdio.h>
#include <omp.h>

#include "../utilities/check.h"
#include "../utilities/utilities.h"

// enable tests
#define FULL      0
#define FULL_ZERO 1  /* use zero ptrs */
#define FULL_S    0  /* need struct support */
#define OFFSET    0 
#define OFFSET_S  0  /* need struct support */

#define N (992)

#define INIT() INIT_LOOP(N, {A[i] = 0; C[i] = 1; D[i] = i; E[i] = -i; s1.A[i] = 0; s1.C[i] = 1; s1.D[i] = i; s1.E[i] = -i; })

typedef struct S {
  double A[N], B[N], C[N], D[N], E[N];
  double *pA, *pB, *pC, *pD, *pE;
} S;

int main(void){
  #if FULL !=0 && FULL_ZERO != 0 && FULL_S != 0 && OFFSET != 0 && OFFSET_S != 0
    check_offloading();
  #endif

  int fail;
  double A[N], B[N], C[N], D[N], E[N];
  double *pA, *pB, *pC, *pD, *pE;
  S s1;

  // map ptrs
  pA = &A[0];
  pB = &B[0];
  pC = &C[0];
  pD = &D[0];
  pE = &E[0];
  s1.pA = &s1.A[0];
  s1.pB = &s1.B[0];
  s1.pC = &s1.C[0];
  s1.pD = &s1.D[0];
  s1.pE = &s1.E[0];

  
#if FULL
  //
  // Test: Execute on device (full extend)
  //
  INIT();
  #pragma omp target device(1) map(from: pA[0:N]) map(to: pC[0:N]) map(pD[0:N]) 
  {
    #pragma omp parallel for schedule(static,1)
    for (int i = 0; i < 992; i++)
      pA[i] = pC[i] + pD[i] + omp_is_initial_device();
  }
  // CHECK: Succeeded
  fail = 0;
  VERIFY(0, N, A[i], (double)(i+1));
  if (fail) {
    printf ("Test full extent: Failed\n");
  } else {
    printf ("Test full extent: Succeeded\n");
  }
#endif

#if FULL_ZERO
  //
  // Test: Execute on device (full extend)
  //
  INIT();
  #pragma omp target data map(from: pA[0:N]) map(to: pC[0:N]) map(pD[0:N]) device(1) 
  {
    #pragma omp target device(1) // implicit zero ptr
    {
      #pragma omp parallel for schedule(static,1) 
      for (int i = 0; i < 992; i++)
        pA[i] = pC[i] + pD[i] + omp_is_initial_device();
    }
  }
  // CHECK: Succeeded
  fail = 0;
  VERIFY(0, N, A[i], (double)(i+1));
  if (fail) {
    printf ("Test full extent with zero length ptrs: Failed\n");
  } else {
    printf ("Test full extent with zero length ptrs: Succeeded\n");
  }
#endif

#if FULL_S
  //
  // Test: Execute on device (full extend
  //
  INIT();
  #pragma omp target device(1) map(s1) map(from: s1.pA[0:N])  map(to: s1.pC[0:N]) map(s1.pD[0:N])
  {
    #pragma omp parallel for schedule(static,1)
    for (int i = 0; i < 992; i++)
      s1.pA[i] = s1.pC[i] + s1.pD[i] + omp_is_initial_device();
  }
  // CHECK: Succeeded
  fail = 0;
  VERIFY(0, N, s1.A[i], (double)(i+1));
  if (fail) {
    printf ("Test full extent struct: Failed\n");
  } else {
    printf ("Test full extent struct: Succeeded\n");
  }
#endif

#if OFFSET
  //
  // Test: Execute on device (with offsets)
  //
  pA = pA - 100;
  pC = pC - 200;
  pD = pD - 300;
  INIT();
  #pragma omp target device(1) map(from: pA[100:N]) map(to: pC[200:N]) map(pD[300:N]) 
  {
    #pragma omp parallel for schedule(static,1)
    for (int i = 0; i < 992; i++)
      pA[i] = pC[i] + pD[i] + omp_is_initial_device();
  }
  // CHECK: Succeeded
  fail = 0;
  VERIFY(0, N, A[i], (double)(i+1));
  if (fail) {
    printf ("Test3: Failed\n");
  } else {
    printf ("Test3: Succeeded\n");
  }
#endif

#if OFFSET_S
  //
  // Test: Execute on device (full extend
  //
  s1.pA = s1.pA - 100;
  s1.pC = s1.pC - 200;
  s1.pD = s1.pD - 300;
  INIT();
  #pragma omp target device(1) map(s1) map(from: s1.pA[100:N])  map(to: s1.pC[200:N]) map(s1.pD[300:N])
  {
    #pragma omp parallel for schedule(static,1)
    for (int i = 0; i < 992; i++)
      s1.pA[i] = s1.pC[i] + s1.pD[i] + omp_is_initial_device();
  }
  // CHECK: Succeeded
  fail = 0;
  VERIFY(0, N, s1.A[i], (double)(i+1));
  if (fail) {
    printf ("Test full extent struct: Failed\n");
  } else {
    printf ("Test full extent struct: Succeeded\n");
  }
#endif

  return 0;
}

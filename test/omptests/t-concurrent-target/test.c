#include <stdio.h>
#include <omp.h>

#include "../utilities/check.h"
#include "../utilities/utilities.h"

// enable tests
#define CHECK              1
#define PAR_A              1
#define PAR_P              1
#define PAR_1_DATA_A       1
#define PAR_T_DATA_A       1
#define PAR_TOFROM_A       1
#define PAR_TOALL_FROM_A   1

#define N 1024   /* data */
#define T 8      /* num threads */
#define M (N/T)  /* data per thread */

#define INIT() INIT_LOOP(N, {A[i] = 0; C[i] = 1; D[i] = i; E[i] = -i; })

int main(void){
#if CHECK
    check_offloading();
#endif

  int fail;
  double A[N], B[N], C[N], D[N], E[N];
  double *pA, *pB, *pC, *pD, *pE;
  int t;

  // map ptrs
  pA = &A[0];
  pB = &B[0];
  pC = &C[0];
  pD = &D[0];
  pE = &E[0];

#if PAR_A
  INIT();
  // each thread compute one quarter
  #pragma omp parallel for num_threads(T) schedule(static, 1)
  for(t=0; t<T; t++) {
    int i0 = t*M;
    int i1 = (t+1)*M;
    #pragma omp target map(A[i0:M], C[i0:M], D[i0:M])
    {
      for(int i = i0; i < i1; i++)
        A[i] = C[i] + D[i] + 1;
    }
  }
  fail = 0;
  VERIFY(0, N, A[i], (double)(i+2));
  if (fail) {
    printf ("Test PAR_A: Failed\n");
  } else {
    printf ("Test PAR_A: Succeeded\n");
  }
#endif

#if PAR_P
  INIT();
  // each thread compute one quarter
  #pragma omp parallel for num_threads(T) schedule(static, 1)
  for(t=0; t<T; t++) {
    int i0 = t*M;
    int i1 = (t+1)*M;
    #pragma omp target map(pA[i0:M], pC[i0:M], pD[i0:M])
    {
      for(int i = i0; i < i1; i++)
        pA[i] = pC[i] + pD[i] + 1;
    }
  }
  fail = 0;
  VERIFY(0, N, A[i], (double)(i+2));
  if (fail) {
    printf ("Test PAR_P: Failed\n");
  } else {
    printf ("Test PAR_P: Succeeded\n");
  }
#endif

#if PAR_1_DATA_A
  INIT();
  // each thread compute one quarter
  #pragma omp target data map(A, B, C)
  {
    #pragma omp parallel for num_threads(T) schedule(static, 1)
    for(t=0; t<T; t++) {
      int i0 = t*M;
      int i1 = (t+1)*M;
      #pragma omp target map(A[i0:M], C[i0:M], D[i0:M])
      {
        for(int i = i0; i < i1; i++)
          A[i] = C[i] + D[i] + 1;
      }
    }
  }
  fail = 0;
  VERIFY(0, N, A[i], (double)(i+2));
  if (fail) {
    printf ("Test PAR_1_DATA_A: Failed\n");
  } else {
    printf ("Test PAR_1_DATA_A: Succeeded\n");
  }
#endif


#if PAR_T_DATA_A
  INIT();
  // each thread compute one quarter
  #pragma omp parallel for num_threads(T) schedule(static, 1)
  for(t=0; t<T; t++) {
    int i0 = t*M;
    int i1 = (t+1)*M;
    #pragma omp target data map(A[i0:M], C[i0:M], D[i0:M]) 
    {
      #pragma omp target map(A[i0:M], C[i0:M], D[i0:M])
      {
        for(int i = i0; i < i1; i++)
          A[i] = C[i] + D[i] + 1;
      }
    }
  }
  fail = 0;
  VERIFY(0, N, A[i], (double)(i+2));
  if (fail) {
    printf ("Test PAR_T_DATA_A: Failed\n");
  } else {
    printf ("Test PAR_T_DATA_A: Succeeded\n");
  }
#endif

#if PAR_TOFROM_A
  INIT();
  // each thread compute one quarter
  #pragma omp parallel for num_threads(T) schedule(static, 1)
  for(t=0; t<T; t++) {
    int i0 = t*M;
    int i1 = (t+1)*M;
    #pragma omp target map(from: A[i0:M]) map(to: C[i0:M], D[i0:M])
    {
      for(int i = i0; i < i1; i++)
        A[i] = C[i] + D[i] + 1;
    }
  }
  fail = 0;
  VERIFY(0, N, A[i], (double)(i+2));
  if (fail) {
    printf ("Test PAR_TOFROM_A: Failed\n");
  } else {
    printf ("Test PAR_TOFROM_A: Succeeded\n");
  }
#endif

#if PAR_TOALL_FROM_A  
  INIT();
  // each thread compute one quarter
  // copy the whole to data... only the first one should move it
  #pragma omp parallel for num_threads(T) schedule(static, 1)
  for(t=0; t<T; t++) {
    int i0 = t*M;
    int i1 = (t+1)*M;
    #pragma omp target map(from: A[i0:M]) map(to: C, D)
    {
      for(int i = i0; i < i1; i++)
        A[i] = C[i] + D[i] + 1;
    }
  }
  fail = 0;
  VERIFY(0, N, A[i], (double)(i+2));
  if (fail) {
    printf ("Test PAR_TOALL_FROM_A: Failed\n");
  } else {
    printf ("Test PAR_TOALL_FROM_A: Succeeded\n");
  }
#endif


  return 0;
}

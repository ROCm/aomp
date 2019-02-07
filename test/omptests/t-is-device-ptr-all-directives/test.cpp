#include <stdio.h>
#include <omp.h>

#define N 2048

int main(void) {

  int A[N];
  int *Aptr = &A[0];
  
  for (auto &a : A)
    a = 0;
    
  #pragma omp target data map(tofrom: Aptr[:N]) use_device_ptr(Aptr)
  { 
    #pragma omp target is_device_ptr(Aptr)
    for (int i=0; i<N; ++i)
      A[i]++;
      
    #pragma omp target parallel is_device_ptr(Aptr)
    if (omp_get_thread_num() == 0)
      for (int i=0; i<N; ++i)
        A[i]++;
      
    #pragma omp target parallel for is_device_ptr(Aptr)
    for (int i=0; i<N; ++i)
      A[i]++;
      
    #pragma omp target parallel for simd is_device_ptr(Aptr)
    for (int i=0; i<N; ++i)
      A[i]++;
      
    #pragma omp target simd is_device_ptr(Aptr)
    for (int i=0; i<N; ++i)
      A[i]++;
      
    #pragma omp target teams is_device_ptr(Aptr)
    if (omp_get_team_num() == 0)
      for (int i=0; i<N; ++i)
        A[i]++;
        
    #pragma omp target teams distribute is_device_ptr(Aptr)
    for (int i=0; i<N; ++i)
      A[i]++;
      
    #pragma omp target teams distribute simd is_device_ptr(Aptr)
    for (int i=0; i<N; ++i)
      A[i]++;
      
    #pragma omp target teams distribute parallel for is_device_ptr(Aptr)
    for (int i=0; i<N; ++i)
      A[i]++;
      
    #pragma omp target teams distribute parallel for simd is_device_ptr(Aptr)
    for (int i=0; i<N; ++i)
      A[i]++;
      
  }
  
  for (int i=0; i<N; ++i)
    if(A[i] != 10) {
      printf("Error! %d != %d\n", A[i], 10);
      return 1;
    }
  
  printf("Success\n");
  return 0;
}

#if 0
#include <stdio.h>
#include <omp.h>

#include "./utilities/check.h"
#include "./utilities/utilities.h"

#define TRIALS (1)

#define N (992)

#define INIT() INIT_LOOP(N, {C[i] = 1; D[i] = i; E[i] = -i;})

#define ZERO(X) ZERO_ARRAY(N, X)

int main(void) {
  check_offloading();

  double A[N], B[N], C[N], D[N], E[N];
  int fail = 0;

  INIT();
  
  int ten = 10;
  int chunkSize = 512/ten;

  
  // ****************************
  // Series 3: with ds attributes
  // ****************************
  // DS currently failing in the compiler with asserts (bug #T158)

  //
  // Test: private
  //
  ZERO(A); ZERO(B);
  double p = 2.0, q = 4.0;
#pragma omp target teams distribute private(p,q) num_teams(256)
    for(int i = 0 ; i < N ; i++) {
      p = 2;
      q = 3;
      A[i] += p;
      B[i] += q;
    } 

  for(int i = 0 ; i < N ; i++) {
    if (A[i] != TRIALS*2) {
      printf("Error at A[%d], h = %lf, d = %lf\n", i, (double) 2, A[i]);
      fail = 1;
    }
    if (B[i] != TRIALS*3) {
      printf("Error at B[%d], h = %lf, d = %lf\n", i, (double) 3, B[i]);
      fail = 1;
    }
  }
  if(fail) printf("Failed\n");
  else printf("Succeeded\n");

  return 0;
}
#endif


#include <stdio.h>
#include <omp.h>

#include "../utilities/check.h"
#include "../utilities/utilities.h"

#define TRIALS (1)

#define N (992)

#define INIT() INIT_LOOP(N, {C[i] = 1; D[i] = i; E[i] = -i;})

#define ZERO(X) ZERO_ARRAY(N, X)

int check_results(double* A){
  for (int i = 0 ; i < N ; i++){
    if (A[i] != TRIALS){
      printf("Error at %d, h = %lf, d = %lf\n", i, (double) TRIALS, A[i]);
      return 0;
    }
  }
  return 1;
}

int check_results_priv(double *A, double *B){
  for(int i = 0 ; i < N ; i++) {
    if (A[i] != TRIALS*3) {
      printf("Error at A[%d], h = %lf, d = %lf\n", i, (double) TRIALS*2, A[i]);
      return 0;
    }
    if (B[i] != TRIALS*7) {
      printf("Error at B[%d], h = %lf, d = %lf\n", i, (double) TRIALS*3, B[i]);
      return 0;
    }
  }
  return 1;
}

#define CODE() \
  ZERO(A); \
  success = 0; \
  for (int t = 0 ; t < TRIALS ; t++) { \
    _Pragma("omp target") \
    _Pragma("omp teams distribute simd CLAUSES") \
    for (int i = 0 ; i < N ; i++){ \
      A[i] += C[i]; \
    } \
  } \
  success += check_results(A); \
  if (success == expected) \
    printf("Succeeded\n");

#define CODE_PRIV() \
  ZERO(A); \
  ZERO(B); \
  p = 2.0; \
  q = 4.0; \
  success = 0; \
  for (int t = 0 ; t < TRIALS ; t++) { \
    _Pragma("omp target") \
    _Pragma("omp teams distribute simd CLAUSES") \
    for (int i = 0 ; i < N ; i++){ \
      p = 3; \
      q = 7; \
      A[i] += p; \
      B[i] += q; \
    } \
  } \
  success += check_results_priv(A, B); \
  if (success == expected) \
    printf("Succeeded\n");

int main(void) {
  check_offloading();

  double A[N], B[N], C[N], D[N], E[N];
  int fail = 0;
  int expected = 1;
  int success = 0;
  int chunkSize;
  double p = 2.0, q = 4.0;
  int nte, tl, blockSize;

  INIT();

  // **************************
  // Series 1: no dist_schedule
  // **************************

  //
  // Test: #iterations == #teams
  //
  printf("iterations = teams\n");
  #define CLAUSES num_teams(992)
  CODE()
  #undef CLAUSES

  printf("iterations > teams\n");
  #define CLAUSES num_teams(256)
  CODE()
  #undef CLAUSES

  printf("iterations < teams\n");
  #define CLAUSES num_teams(1024)
  CODE()
  #undef CLAUSES

  printf("num_teams(512) dist_schedule(static,1)\n");
  #define CLAUSES num_teams(512) dist_schedule(static, 1)
  CODE()
  #undef CLAUSES

  printf("num_teams(512) dist_schedule(static,512)\n");
  #define CLAUSES num_teams(512) dist_schedule(static, 512)
  CODE()
  #undef CLAUSES

  printf("num_teams(512) dist_schedule(static, chunkSize)\n");
  chunkSize = N / 10;
  #define CLAUSES num_teams(512) dist_schedule(static, chunkSize)
  CODE()
  #undef CLAUSES

  printf("num_teams(1024) dist_schedule(static, chunkSize)\n");
  chunkSize = N / 10;
  #define CLAUSES num_teams(1024) dist_schedule(static, chunkSize)
  CODE()
  #undef CLAUSES

  printf("num_teams(1024) dist_schedule(static, 1)\n");
  #define CLAUSES num_teams(1024) dist_schedule(static, 1)
  CODE()
  #undef CLAUSES

  printf("num_teams(3) dist_schedule(static, 1)\n");
  #define CLAUSES num_teams(3) dist_schedule(static, 1)
  CODE()
  #undef CLAUSES

  printf("num_teams(3) dist_schedule(static, 3)\n");
  #define CLAUSES num_teams(3) dist_schedule(static, 3)
  CODE()
  #undef CLAUSES

  printf("num_teams(10) dist_schedule(static, 99)\n");
  #define CLAUSES num_teams(10) dist_schedule(static, 99)
  CODE()
  #undef CLAUSES

  printf("num_teams(256) dist_schedule(static, 992)\n");
  #define CLAUSES num_teams(256) dist_schedule(static, 992)
  CODE()
  #undef CLAUSES

#if 0
  printf("num_teams(256) private(p,q)\n");
  #define CLAUSES num_teams(256) private(p,q)
  CODE_PRIV()
  #undef CLAUSES
#endif

  //
  // Test: firstprivate
  //

#if 0
  printf("num_teams(64) firstprivate(p, q)\n");
  ZERO(A); ZERO(B);
  p = 2.0, q = 4.0;
  for (int t = 0 ; t < TRIALS ; t++) {
    #pragma omp target // implicit firstprivate for p and q, their initial values being 2 and 4 for each target invocation
    #pragma omp teams distribute simd num_teams(64) firstprivate(p, q)
    for(int i = 0 ; i < 128 ; i++) { // 2 iterations for each team
      p += 3.0;  // p and q are firstprivate to the team, and as such incremented twice (2 iterations per team)
      q += 7.0;
      A[i] += p;
      B[i] += q;
    }
  }
  for(int i = 0 ; i < 128 ; i++) {
    if (i % 2 == 0) {
      if (A[i] != (2.0+3.0)*TRIALS) {
      	printf("Error at A[%d], h = %lf, d = %lf\n", i, (double) (2.0+3.0)*TRIALS, A[i]);
      	fail = 1;
      }
      if (B[i] != (4.0+7.0)*TRIALS) {
      	printf("Error at B[%d], h = %lf, d = %lf\n", i, (double) (4.0+7.0)*TRIALS, B[i]);
      	fail = 1;
      }
    } else {
      if (A[i] != (2.0+3.0*2)*TRIALS) {
      	printf("Error at A[%d], h = %lf, d = %lf\n", i, (double) (2.0+3.0*2)*TRIALS, A[i]);
      	fail = 1;
      }
      if (B[i] != (4.0+7.0*2)*TRIALS) {
      	printf("Error at B[%d], h = %lf, d = %lf\n", i, (double) (4.0+7.0*2)*TRIALS, B[i]);
      	fail = 1;
      }
    }
  }
  if(fail) printf("Failed\n");
  else printf("Succeeded\n");
#endif

  //
  // Test: lastprivate
  //

#if 0
  printf("num_teams(10) lastprivate(lastpriv)\n");
  success = 0;
  int lastpriv = -1;
  #pragma omp target map(tofrom:lastpriv)
  #pragma omp teams distribute simd num_teams(10) lastprivate(lastpriv)
  for(int i = 0 ; i < omp_get_num_teams() ; i++)
    lastpriv = omp_get_team_num();

  if(lastpriv != 9) {
    printf("lastpriv value is %d and should have been %d\n", lastpriv, 9);
    fail = 1;
  }

  if(fail) printf("Failed\n");
  else printf("Succeeded\n");
#endif

  // // ***************************
  // // Series 4: with parallel for
  // // ***************************

  //
  // Test: simple blocking loop
  //
  printf("num_teams(nte) thread_limit(tl) with parallel for innermost\n");
  success = 0;
  ZERO(A); ZERO(B);
  nte = 32;
  tl = 64;
  blockSize = tl;

  for (int t = 0 ; t < TRIALS ; t++) {
    #pragma omp target
    #pragma omp teams distribute simd num_teams(nte) thread_limit(tl)
    for(int j = 0 ; j < 256 ; j += blockSize) {
      for(int i = j ; i < j+blockSize; i++) {
        A[i] += B[i] + C[i];
      }
    }
  }
  for(int i = 0 ; i < 256 ; i++) {
    if (A[i] != TRIALS) {
      printf("Error at A[%d], h = %lf, d = %lf\n", i, (double) (2.0+3.0)*TRIALS, A[i]);
      fail = 1;
    }
  }

  if(fail) printf("Failed\n");
  else printf("Succeeded\n");

  //
  // Test: blocking loop where upper bound is not a multiple of tl*nte
  //

  printf("num_teams(nte) thread_limit(tl) with parallel for innermost\n");
  success = 0;
  ZERO(A); ZERO(B);
  nte = 32;
  tl = 64;
  blockSize = tl;

  for (int t = 0 ; t < TRIALS ; t++) {
    #pragma omp target
    #pragma omp teams distribute simd num_teams(nte) thread_limit(tl)
    for(int j = 0 ; j < 510 ; j += blockSize) {
      int ub = (j+blockSize < 510) ? (j+blockSize) : 512;
      for(int i = j ; i < ub; i++) {
        A[i] += B[i] + C[i];
      }
    }
  }
  for(int i = 0 ; i < 256 ; i++) {
    if (A[i] != TRIALS) {
      printf("Error at A[%d], h = %lf, d = %lf\n", i, (double) (2.0+3.0)*TRIALS, A[i]);
      fail = 1;
    }
  }

  if(fail) printf("Failed\n");
  else printf("Succeeded\n");

  // **************************
  // Series 5: collapse
  // **************************

  //
  // Test: 2 loops
  //

  printf("num_teams(512) collapse(2)\n");
  success = 0;
  double * S = malloc(N*N*sizeof(double));
  double * T = malloc(N*N*sizeof(double));
  double * U = malloc(N*N*sizeof(double));
  for (int i = 0 ; i < N ; i++)
    for (int j = 0 ; j < N ; j++)
    {
      S[i*N+j] = 0.0;
      T[i*N+j] = 1.0;
      U[i*N+j] = 2.0;
    }

  for (int t = 0 ; t < TRIALS ; t++) {
    #pragma omp target map(tofrom:S[:N*N]), map(to:T[:N*N],U[:N*N])
    #pragma omp teams distribute simd num_teams(512) collapse(2)
    for (int i = 0 ; i < N ; i++)
      for (int j = 0 ; j < N ; j++)
        S[i*N+j] += T[i*N+j] + U[i*N+j];  // += 3 at each t
  }
  for (int i = 0 ; i < N ; i++)
    for (int j = 0 ; j < N ; j++)
      if (S[i*N+j] != TRIALS*3.0) {
        printf("Error at (%d,%d), h = %lf, d = %lf\n", i, j, (double) TRIALS*3.0, S[i*N+j]);
        fail = 1;
      }
  if(fail) printf("Failed\n");
  else printf("Succeeded\n");

  //
  // Test: 3 loops
  //

  printf("num_teams(512) collapse(3)\n");
  success = 0;
  int M = N/8;
  double * V = malloc(M*M*M*sizeof(double));
  double * Z = malloc(M*M*M*sizeof(double));
  for (int i = 0 ; i < M ; i++)
    for (int j = 0 ; j < M ; j++)
      for (int k = 0 ; k < M ; k++)
      {
        V[i*M*M+j*M+k] = 2.0;
        Z[i*M*M+j*M+k] = 3.0;
      }

  for (int t = 0 ; t < TRIALS ; t++) {
    #pragma omp target map(tofrom:V[:M*M*M]), map(to:Z[:M*M*M])
    #pragma omp teams distribute simd num_teams(512) collapse(3)
    for (int i = 0 ; i < M ; i++)
      for (int j = 0 ; j < M ; j++)
        for (int k = 0 ; k < M ; k++)
          V[i*M*M+j*M+k] += Z[i*M*M+j*M+k];  // += 3 at each t
  }
  for (int i = 0 ; i < M ; i++)
    for (int j = 0 ; j < M ; j++)
      for (int k = 0 ; k < M ; k++)
        if (V[i*M*M+j*M+k] != 2.0+TRIALS*3.0) {
          printf("Error at (%d,%d), h = %lf, d = %lf\n", i, j, (double) TRIALS*3.0, V[i*M*M+j*M+k]);
          fail = 1;
        }
  if(fail) printf("Failed\n");
  else printf("Succeeded\n");

  return 0;
}


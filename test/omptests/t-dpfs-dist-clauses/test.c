
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
  int fail = 0;

  INIT();

  // **************************
  // Series 1: no dist_schedule
  // **************************

  //
  // Test: #iterations == #teams
  //
  ZERO(A);
  for (int t = 0 ; t < TRIALS ; t++) {
    #pragma omp target
    #pragma omp teams num_teams(512)
    #pragma omp distribute parallel for simd
    for (int i = 0 ; i < 512 ; i++)
    {
      A[i] += C[i]; // += 1 per position
    }
  }
  for (int i = 0 ; i < 512 ; i++)
    if (A[i] != TRIALS) {
      printf("Error at %d, h = %lf, d = %lf\n", i, (double) TRIALS, A[i]);
      fail = 1;
    }
  if(fail) printf("Failed\n");
  else printf("Succeeded\n");

  //
  // Test: #iterations > #teams
  //
  ZERO(A);
  for (int t = 0 ; t < TRIALS ; t++) {
    #pragma omp target
    #pragma omp teams num_teams(256)
    #pragma omp distribute parallel for simd
    for (int i = 0 ; i < 500 ; i++)
    {
      A[i] += C[i]; // += 1 per position
    }
  }
  for (int i = 0 ; i < 500 ; i++)
    if (A[i] != TRIALS) {
      printf("Error at %d, h = %lf, d = %lf\n", i, (double) TRIALS, A[i]);
      fail = 1;
    }
  if(fail) printf("Failed\n");
  else printf("Succeeded\n");

  //
  // Test: #iterations < #teams
  //
  ZERO(A);
  for (int t = 0 ; t < TRIALS ; t++) {
    #pragma omp target
    #pragma omp teams num_teams(256)
    #pragma omp distribute parallel for simd
    for (int i = 0 ; i < 123 ; i++)
    {
      A[i] += C[i]; // += 1 per position
    }
  }
  for (int i = 0 ; i < 123 ; i++)
    if (A[i] != TRIALS) {
      printf("Error at %d, h = %lf, d = %lf\n", i, (double) TRIALS, A[i]);
      fail = 1;
    }
  if(fail) printf("Failed\n");
  else printf("Succeeded\n");

  // ****************************
  // Series 2: with dist_schedule
  // ****************************

  //
  // Test: #iterations == #teams, dist_schedule(1)
  //
  ZERO(A);
  for (int t = 0 ; t < TRIALS ; t++) {
    #pragma omp target
    #pragma omp teams num_teams(512)
    #pragma omp distribute parallel for simd dist_schedule(static,1)
    for (int i = 0 ; i < 512 ; i++)
    {
      A[i] += C[i]; // += 1 per position
    }
  }
  for (int i = 0 ; i < 512 ; i++)
    if (A[i] != TRIALS) {
      printf("Error at %d, h = %lf, d = %lf\n", i, (double) TRIALS, A[i]);
      fail = 1;
    }
  if(fail) printf("Failed\n");
  else printf("Succeeded\n");

  //
  // Test: #iterations == #teams, dist_schedule(#iterations)
  //
  ZERO(A);
  for (int t = 0 ; t < TRIALS ; t++) {
    #pragma omp target
    #pragma omp teams num_teams(512)
    #pragma omp distribute parallel for simd dist_schedule(static,512)
    for (int i = 0 ; i < 512 ; i++)
    {
      A[i] += C[i]; // += 1 per position
    }
  }
  for (int i = 0 ; i < 512 ; i++)
    if (A[i] != TRIALS) {
      printf("Error at %d, h = %lf, d = %lf\n", i, (double) TRIALS, A[i]);
      fail = 1;
    }
  if(fail) printf("Failed\n");
  else printf("Succeeded\n");

  //
  // Test: #iterations == #teams, dist_schedule(#iterations/10), variable chunk size
  //
  ZERO(A);
  int ten = 10;
  int chunkSize = 512/ten;
  for (int t = 0 ; t < TRIALS ; t++) {
    #pragma omp target
    #pragma omp teams num_teams(512)
    #pragma omp distribute parallel for simd dist_schedule(static,chunkSize)
    for (int i = 0 ; i < 512 ; i++)
    {
      A[i] += C[i]; // += 1 per position
    }
  }
  for (int i = 0 ; i < 512 ; i++)
    if (A[i] != TRIALS) {
      printf("Error at %d, h = %lf, d = %lf\n", i, (double) TRIALS, A[i]);
      fail = 1;
    }
  if(fail) printf("Failed\n");
  else printf("Succeeded\n");

  //
  // Test: #iterations > #teams, dist_schedule(1)
  //
    ZERO(A);
  for (int t = 0 ; t < TRIALS ; t++) {
    #pragma omp target
    #pragma omp teams num_teams(256)
    #pragma omp distribute parallel for simd dist_schedule(static,1)
    for (int i = 0 ; i < 500 ; i++)
    {
      A[i] += C[i]; // += 1 per position
    }
  }
  for (int i = 0 ; i < 500 ; i++)
    if (A[i] != TRIALS) {
      printf("Error at %d, h = %lf, d = %lf\n", i, (double) TRIALS, A[i]);
      fail = 1;
    }
  if(fail) printf("Failed\n");
  else printf("Succeeded\n");

  //
  // Test: #iterations > #teams, dist_schedule(#iterations)
  //
  ZERO(A);
  for (int t = 0 ; t < TRIALS ; t++) {
    #pragma omp target
    #pragma omp teams num_teams(256)
    #pragma omp distribute parallel for simd dist_schedule(static,500)
    for (int i = 0 ; i < 500 ; i++)
    {
      A[i] += C[i]; // += 1 per position
    }
  }
  for (int i = 0 ; i < 500 ; i++)
    if (A[i] != TRIALS) {
      printf("Error at %d, h = %lf, d = %lf\n", i, (double) TRIALS, A[i]);
      fail = 1;
    }
  if(fail) printf("Failed\n");
  else printf("Succeeded\n");

  //
  // Test: #iterations > #teams, dist_schedule(#iterations/10), variable chunk size
  //
  ZERO(A);
  ten = 10;
  chunkSize = 500/ten;
  for (int t = 0 ; t < TRIALS ; t++) {
    #pragma omp target
    #pragma omp teams num_teams(256)
    #pragma omp distribute parallel for simd dist_schedule(static,chunkSize)
    for (int i = 0 ; i < 500 ; i++)
    {
      A[i] += C[i]; // += 1 per position
    }
  }
  for (int i = 0 ; i < 500 ; i++)
    if (A[i] != TRIALS) {
      printf("Error at %d, h = %lf, d = %lf\n", i, (double) TRIALS, A[i]);
      fail = 1;
    }
  if(fail) printf("Failed\n");
  else printf("Succeeded\n");

  //
  // Test: #iterations < #teams, dist_schedule(1)
  //
  ZERO(A);
  for (int t = 0 ; t < TRIALS ; t++) {
    #pragma omp target
    #pragma omp teams num_teams(256)
#pragma omp distribute parallel for simd dist_schedule(static,1)
    for (int i = 0 ; i < 123 ; i++)
    {
      A[i] += C[i]; // += 1 per position
    }
  }
  for (int i = 0 ; i < 123 ; i++)
    if (A[i] != TRIALS) {
      printf("Error at %d, h = %lf, d = %lf\n", i, (double) TRIALS, A[i]);
      fail = 1;
    }
  if(fail) printf("Failed\n");
  else printf("Succeeded\n");

  //
  // Test: #iterations < #teams, dist_schedule(#iterations)
  //
  ZERO(A);
  for (int t = 0 ; t < TRIALS ; t++) {
    #pragma omp target
    #pragma omp teams num_teams(256)
#pragma omp distribute parallel for simd dist_schedule(static,123)
    for (int i = 0 ; i < 123 ; i++)
    {
      A[i] += C[i]; // += 1 per position
    }
  }
  for (int i = 0 ; i < 123 ; i++)
    if (A[i] != TRIALS) {
      printf("Error at %d, h = %lf, d = %lf\n", i, (double) TRIALS, A[i]);
      fail = 1;
    }
  if(fail) printf("Failed\n");
  else printf("Succeeded\n");

  //
  // Test: #iterations < #teams, dist_schedule(#iterations)
  //
  ZERO(A);
  ten = 10;
  chunkSize = 123/ten;
  for (int t = 0 ; t < TRIALS ; t++) {
    #pragma omp target
    #pragma omp teams num_teams(256)
    #pragma omp distribute parallel for simd dist_schedule(static,chunkSize)
    for (int i = 0 ; i < 123 ; i++)
    {
      A[i] += C[i]; // += 1 per position
    }
  }
  for (int i = 0 ; i < 123 ; i++)
    if (A[i] != TRIALS) {
      printf("Error at %d, h = %lf, d = %lf\n", i, (double) TRIALS, A[i]);
      fail = 1;
    }
  if(fail) printf("Failed\n");
  else printf("Succeeded\n");

  // ****************************
  // Series 3: with ds attributes
  // ****************************

  //
  // Test: private
  //
  ZERO(A); ZERO(B);
  double p = 2.0, q = 4.0;
  for (int t = 0 ; t < TRIALS ; t++) {
    #pragma omp target
    #pragma omp teams num_teams(256)
    {
      #pragma omp distribute parallel for simd private(p,q)
      for(int i = 0 ; i < N ; i++) {
	p = 2;
	q = 3;
	A[i] += p;
	B[i] += q;
      }
    }
  }
  for(int i = 0 ; i < N ; i++) {
    if (A[i] != TRIALS*2) {
      printf("Error at A[%d], h = %lf, d = %lf\n", i, (double) TRIALS*2, A[i]);
      fail = 1;
    }
    if (B[i] != TRIALS*3) {
      printf("Error at B[%d], h = %lf, d = %lf\n", i, (double) TRIALS*3, B[i]);
      fail = 1;
    }
  }
  if(fail) printf("Failed\n");
  else printf("Succeeded\n");

  //
  // Test: firstprivate
  //
  ZERO(A); ZERO(B);
  p = 2.0, q = 4.0;
  for (int t = 0 ; t < TRIALS ; t++) {
#pragma omp target // implicit firstprivate for p and q, their initial values being 2 and 4 for each target invocation
#pragma omp teams num_teams(64)
    {
      #pragma omp distribute simd firstprivate(p,q)
      for(int i = 0 ; i < 128 ; i++) { // 2 iterations for each team
	p += 3.0;  // p and q are firstprivate to the team, and as such incremented twice (2 iterations per team)
	q += 7.0;
	A[i] += p;
	B[i] += q;
      }
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

  //
  // Test: lastprivate
  //

  int lastpriv = -1;
#pragma omp target map(tofrom:lastpriv)
#pragma omp teams num_teams(10)
#pragma omp distribute parallel for simd lastprivate(lastpriv)
  for(int i = 0 ; i < omp_get_num_teams() ; i++)
    lastpriv = omp_get_team_num();

  if(lastpriv != 9) {
    printf("lastpriv value is %d and should have been %d\n", lastpriv, 9);
    fail = 1;
  }

  if(fail) printf("Failed\n");
  else printf("Succeeded\n");

  // **************************
  // Series 4: collapse
  // **************************

  //
  // Test: 2 loops
  //
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
    #pragma omp teams num_teams(512)
    #pragma omp distribute parallel for simd collapse(2)
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
    #pragma omp teams num_teams(512)
    #pragma omp distribute parallel for simd collapse(3)
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


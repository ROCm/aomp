
#include <stdio.h>
#include <omp.h>

#include "../utilities/check.h"
#include "../utilities/utilities.h"

#define HOST_MAX_TEAMS 128

#define TRIALS (1)

#define N (992)

#define INIT() INIT_LOOP(N, {C[i] = 1; D[i] = i; E[i] = -i;})

#define ZERO(X) ZERO_ARRAY(N, X)

int main(void) {
  check_offloading();

  double A[N], B[N], C[N], D[N], E[N];
  double * pA = malloc(N*sizeof(double));
  int fail = 0;

  INIT();

  //
  // Test: if clause
  //
  ZERO(A);
  int num_teams = omp_is_initial_device() ? HOST_MAX_TEAMS : 512;
  // the number of teams started is implementation dependent
  int actual_teams = -1;
  for (int t = 0 ; t < TRIALS ; t++) {
#pragma omp target teams if(0) map(tofrom:actual_teams)
    {
      if(omp_get_team_num() == 0)
	actual_teams = omp_get_num_teams();
      A[omp_get_team_num()] += omp_get_team_num();
    }
  }
  for (int i = 0 ; i < actual_teams ; i++)
    if (A[i] != i*TRIALS) {
      printf("Error at %d, h = %lf, d = %lf\n", i, (double) i*TRIALS, A[i]);
      fail = 1;
    }
  if(fail) printf("Failed\n");
  else printf("Succeeded\n");

  //
  // Test: device clause
  //
  ZERO(A);
  num_teams = omp_is_initial_device() ? HOST_MAX_TEAMS : 512;
  for (int t = 0 ; t < TRIALS ; t++) {
#pragma omp target teams device(0) map(tofrom:actual_teams)
    {
      if(omp_get_team_num() == 0)
	actual_teams = omp_get_num_teams();
      A[omp_get_team_num()] += omp_get_team_num();
    }
  }
  for (int i = 0 ; i < actual_teams ; i++)
    if (A[i] != i*TRIALS) {
      printf("Error at %d, h = %lf, d = %lf\n", i, (double) i*TRIALS, A[i]);
      fail = 1;
    }
  if(fail) printf("Failed\n");
  else printf("Succeeded\n");

  //
  // Test: map clause
  //
  ZERO(pA);
  num_teams = omp_is_initial_device() ? HOST_MAX_TEAMS : 512;
  for (int t = 0 ; t < TRIALS ; t++) {
  #pragma omp target teams map(pA[:N]) map(tofrom:actual_teams)
    {
      if(omp_get_team_num() == 0)
	actual_teams = omp_get_num_teams();
      pA[omp_get_team_num()] += omp_get_team_num();
    }
  }
  for (int i = 0 ; i < actual_teams ; i++)
    if (pA[i] != i*TRIALS) {
      printf("Error at %d, h = %lf, d = %lf\n", i, (double) i*TRIALS, pA[i]);
      fail = 1;
    }
  if(fail) printf("Failed\n");
  else printf("Succeeded\n");

  //
  // Test: num_teams and omp_get_team_num()
  //
  ZERO(A);
  num_teams = omp_is_initial_device() ? HOST_MAX_TEAMS : 512;
  for (int t = 0 ; t < TRIALS ; t++) {
    #pragma omp target teams num_teams(num_teams)
    {
      A[omp_get_team_num()] += omp_get_team_num();
    }
  }
  for (int i = 0 ; i < num_teams ; i++)
    if (A[i] != i*TRIALS) {
      printf("Error at %d, h = %lf, d = %lf\n", i, (double) i*TRIALS, A[i]);
      fail = 1;
    }
  if(fail) printf("Failed\n");
  else printf("Succeeded\n");

  //
  // Test: thread_limit and omp_get_thread_num()
  //
  ZERO(A);
  fail = 0;
  int num_threads = omp_is_initial_device() ? HOST_MAX_TEAMS : 256;
  for (int t = 0 ; t < TRIALS ; t++) {
    #pragma omp target teams num_teams(1) thread_limit(num_threads)
    #pragma omp parallel
    {
      int tid = omp_get_thread_num();
      A[tid] += (double) tid;
    }
  }
  for (int i = 0 ; i < num_threads ; i++)
    if (A[i] != i*TRIALS) {
      printf("Error at %d, h = %lf, d = %lf\n", i, (double) i*TRIALS, A[i]);
      fail = 1;
    }
  if(fail) printf("Failed\n");
  else printf("Succeeded\n");

  //
  // Test: if statement in teams region
  //
  ZERO(A);
  fail = 0;
  num_teams = omp_is_initial_device() ? HOST_MAX_TEAMS : 512;
  for (int t = 0 ; t < TRIALS ; t++) {
    #pragma omp target teams num_teams(num_teams)
    {
      if (omp_get_team_num() % 2 == 0) {
  	int teid = omp_get_team_num();
  	A[teid] += (double) 1;
      }
      else {
  	int teid = omp_get_team_num();
  	A[teid] += (double) 2;
      }
    }
  }
  for (int i = 0 ; i < num_teams ; i++) {
    if (i % 2 == 0) {
      if (A[i] != TRIALS) {
  	printf("Error at %d, h = %lf, d = %lf\n", i, (double) TRIALS, A[i]);
  	fail = 1;
      }
    } else
      if (A[i] != 2*TRIALS) {
  	printf("Error at %d, h = %lf, d = %lf\n", i, (double) 2*TRIALS, A[i]);
  	fail = 1;
      }
  }
  if(fail) printf("Failed\n");
  else printf("Succeeded\n");

  /* // */
  /* // Test: num_teams and thread_limit by simulating a distribute pragma */
  /* // */
  /* ZERO(A); */
  /* fail = 0; */
  /* for (int t = 0 ; t < TRIALS ; t++) { */
  /*   #pragma omp target teams num_teams(2) thread_limit(496) */
  /*   { */
  /*     if (omp_get_team_num() == 0) { */
  /* 	#pragma omp parallel */
  /* 	{ */
  /* 	  A[omp_get_team_num()*496+omp_get_thread_num()] += omp_get_thread_num(); */
  /* 	  if(omp_get_thread_num() == 498) printf("teid = %d, tid = %d, accessing %d\n", omp_get_team_num(), omp_get_thread_num(), omp_get_team_num()*496+omp_get_thread_num()); */
  /* 	} */
  /*     } else { */
  /* 	#pragma omp parallel */
  /* 	{ */
  /* 	  if(omp_get_thread_num() == 0) */
  /* 	    printf("teid = %d, tid = %d: A= %lf\n", omp_get_team_num(), omp_get_thread_num(), A[omp_get_team_num()*496+omp_get_thread_num()]); */
  /* 	  A[omp_get_team_num()*496+omp_get_thread_num()] -= omp_get_thread_num(); */
  /* 	  if(omp_get_thread_num() == 0) */
  /* 	    printf("teid = %d, tid = %d: A= %lf\n", omp_get_team_num(), omp_get_thread_num(), A[omp_get_team_num()*496+omp_get_thread_num()]); */
  /* 	} */
  /*     } */
  /*   } */
  /* } */
  /* for (int i = 0 ; i < 992 ; i++) { */
  /*   if (i < 496) { */
  /*     if (A[i] != i*TRIALS) { */
  /* 	printf("Error at %d, h = %lf, d = %lf\n", i, (double) i*TRIALS, A[i]); */
  /* 	fail = 1; */
  /*     } */
  /*   } else if(i >= 496) */
  /*     if (A[i] != -((i-496)*TRIALS)) { */
  /* 	printf("Error at %d, h = %lf, d = %lf\n", i, (double) -((i-496)*TRIALS), A[i]); */
  /* 	fail = 1; */
  /*     } */
  /* } */
  /* if(fail) printf("Failed\n"); */
  /* else printf("Succeeded\n"); */

  //
  // Test: private
  //
  ZERO(A);
  fail = 0;
  int a = 10;
  num_teams = omp_is_initial_device() ? HOST_MAX_TEAMS : 256;
  for (int t = 0 ; t < TRIALS ; t++) {
    #pragma omp target teams num_teams(num_teams) private(a)
    {
      a = omp_get_team_num();
      A[omp_get_team_num()] += a;
    }
  }

  for (int i = 0 ; i < num_teams ; i++)
    if (A[i] != i*TRIALS) {
      printf("Error at %d, h = %lf, d = %lf\n", i, (double) i*TRIALS, A[i]);
      fail = 1;
    }
  if(fail) printf("Failed\n");
  else printf("Succeeded\n");

  //
  // Test: firstprivate
  //  
  ZERO(A);
  fail = 0;
  a = 10;
  num_teams = omp_is_initial_device() ? HOST_MAX_TEAMS : 256;
  for (int t = 0 ; t < TRIALS ; t++) {
    #pragma omp target teams num_teams(num_teams) firstprivate(a)
    {
      a += omp_get_team_num();
      A[omp_get_team_num()] += a;
    }
  }

  for (int i = 0 ; i < num_teams ; i++)
    if (A[i] != 10+i*TRIALS) {
      printf("Error at %d, h = %lf, d = %lf\n", i, (double) (10+i*TRIALS), A[i]);
      fail = 1;
    }
  if(fail) printf("Failed\n");
  else printf("Succeeded\n");
  
  return 0;
}

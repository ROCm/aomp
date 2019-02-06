#include <stdio.h>
#include <omp.h>

#define HOST_MAX_TEAMS 128

#define TRIALS (1)

#define N (992)

int main(void) {

  int A[N], B[N], C[N], D[N], E[N];
  int fail = 0;

  //
  // Test: num_teams and omp_get_team_num()
  //
  for (int i = 0; i < N; i++) {
    A[i] = 0;
    C[i] = 1;
    D[i] = i;
    E[i] = (-1)*i;
  }

  int num_teams = omp_is_initial_device() ? HOST_MAX_TEAMS : 512;
  printf("Using num_teams = %d\n", num_teams);

  for (int t = 0 ; t < TRIALS ; t++) {
    #pragma omp target
    #pragma omp teams num_teams(num_teams)
    {
      A[omp_get_team_num()] += omp_get_team_num();
    }
  }
  for (int i = 0 ; i < num_teams ; i++)
    if (A[i] != i*TRIALS) {
      printf("Error at %d, h = %d, d = %d\n", i, i*TRIALS, A[i]);
      fail = 1;
    }
  if(fail) printf("Failed\n");
  else printf("Succeeded\n");

  return fail;
}


#include <stdlib.h>
#include <stdio.h>
#include <omp.h>

#define N 10000
#define TOTAL_TEAMS 16
int main() {
  int n = N;
  int team_id;
  int team_counts[TOTAL_TEAMS];
  int *a = (int *)malloc(n*sizeof(int));

  for (int i = 0; i < TOTAL_TEAMS; i++)
      team_counts[i] = 0;

  #pragma omp teams distribute num_teams(TOTAL_TEAMS) private(team_id)
  for(int i = 0; i < n; i++) {
    team_id = omp_get_team_num();
    a[i] = i;
    team_counts[team_id]++;
  }
  int err = 0;
  for(int i = 0; i < n; i++) {
    if (a[i] != i) {
      printf("Error at %d: a = %d, should be %d\n", i, a[i], i);
      err++;
      if (err > 10) break;
    }
  }

  for (int i = 0; i < TOTAL_TEAMS; i++) {
      if (team_counts[i] != N/TOTAL_TEAMS) {
          printf("Team id : %d is not shared with equal work. It is shared"
			  " with %d iterations\n", i, team_counts[i]);
	  err++;
      }
  }
  team_id = omp_get_team_num();
  // omp_get_team_num() should return 0 when called outside of teams region.
  if (team_id != 0) {
      printf("omp_get_team_num() : %d expected 0 when omp_get_team_num()"
		      " called outside of teams region\n", team_id);
      err++;
  }
  return err;
}

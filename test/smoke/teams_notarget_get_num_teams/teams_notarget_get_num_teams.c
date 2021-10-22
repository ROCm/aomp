#include <stdlib.h>
#include <stdio.h>
#include <omp.h>

#define N 10000
#define MAX_TEAMS 64
int main() {
  int n = N;
  int team_id, cur_teams;
  int teams_sizes[10] = {1, 2, 4, 8, 16, 32, 64, 128, 256, 512};
  int *a = (int *)malloc(n*sizeof(int));
  int err = 0;

  for (int j = 0; j < 10; j++) {
      cur_teams = 0;
      #pragma omp teams distribute num_teams(teams_sizes[j])
      for (int i = 0; i < n; i++) {
          cur_teams = omp_get_num_teams();
          a[i] = i;
      }
      err = 0;
      for (int i = 0; i < n; i++) {
          if (a[i] != i) {
              printf("Error at %d: a = %d, should be %d\n", i, a[i], i);
              err++;
              if (err > 10) break;
          }
      }
      // If we have bigger number than MAX_TEAMS in num_teams() clause we will
      // get omp_get_num_teams() as MAX_TEAMS.
      if ( ((cur_teams > MAX_TEAMS) && (cur_teams != MAX_TEAMS)) && (cur_teams != teams_sizes[j]) ) {
          printf("omp_get_num_teams() : %d but we tried to set num_teams(%d)\n", cur_teams, teams_sizes[j]);
	  err++;
      }
  }
  return err;
}

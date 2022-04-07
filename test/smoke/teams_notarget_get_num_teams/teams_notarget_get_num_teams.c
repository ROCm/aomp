#include <stdlib.h>
#include <stdio.h>
#include <omp.h>

#define N 10000
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
      // omp_get_num_teams() will always return value less than or equal to
      // value passed in num_teams() clause.
      if ( cur_teams > teams_sizes[j] ) {
          printf("Error : omp_get_num_teams() : %d but we tried to set"
			  " num_teams(%d)\n", cur_teams, teams_sizes[j]);
	  err++;
      }
  }
  cur_teams = omp_get_num_teams();
  // omp_get_num_teams() value should be 1 when outside of teams region.
  if ( cur_teams != 1 ) {
      printf("Error : omp_get_num_teams() : %d but should return 1 when"
		      " outside of teams region.\n", cur_teams);
      err++;
  }
  return err;
}

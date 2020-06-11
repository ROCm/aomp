#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <omp.h>

#define N 128
#define THRESHOLD  127
int main() {
   static float x[N],y[N] __attribute__ ((aligned(64)));
   float s=2.0;
   int return_code = 0 ;

///  How can we set number of teams to number numa domains and what is the mapping?
#pragma omp teams num_teams(32)
{
printf("teams:%d num_teams%d threads:%d  ishost:%d\n",omp_get_num_teams(), omp_get_team_num(),
		omp_get_num_threads(), omp_is_initial_device());
}

   return 0;
}

#include <stdio.h>
#include <string.h>
#include <omp.h>
#define THREADS 2
#define TEAMS 2

int main(){
  int gpu_results[TEAMS];
  int correct_results[TEAMS];
  int actual_num_threads = -1;

  // the runtime is allowed to use <=THREADS in the parallel regions and
  // it actually chooses 1 (new runtime) or 2 (old runtime)
  #pragma omp target teams thread_limit(THREADS) num_teams(TEAMS) \
    map(from:gpu_results, actual_num_threads)
  {
    int dist[THREADS];
    // Uncomment line below to trigger generic kernel before fix was in place
    //dist[0] = 0;
    #pragma omp parallel
    {
      int thread = omp_get_thread_num();
      int team = omp_get_team_num();
      dist[thread] = 0;

      #pragma omp barrier

      dist[thread] += 1;

      #pragma omp barrier

      if(thread == 0) {
	if (team == 0)
	  actual_num_threads = omp_get_num_threads();
        for(int i = 1; i < omp_get_num_threads(); i++)
          dist[0] += dist[i];
        gpu_results[team] = dist[0];
      }
    }
  }

  for(int i = 0; i < TEAMS; i++)
    correct_results[i] = actual_num_threads;
  int status = memcmp(correct_results, gpu_results, TEAMS * sizeof(int));

  if (status != 0) {
    printf("FAIL status %d\n",status);
    return 1;
  }
  if (actual_num_threads > 2) {
    printf("FAIL threads %d\n",actual_num_threads);
    return 1;
  }
  printf("PASS\n");
  return 0;
}

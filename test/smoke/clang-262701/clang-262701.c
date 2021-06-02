#include <stdio.h>
#include <string.h>
#include <omp.h>
#define THREADS 2
#define TEAMS 2

int main(){
  int gpu_results[THREADS];
  int correct_results[THREADS] = {2,2};
  #pragma omp target teams thread_limit(THREADS) num_teams(TEAMS) map(from:gpu_results)
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
        for(int i = 1; i < THREADS; i++)
          dist[0] += dist[i];
        gpu_results[team] = dist[0];
      }
    }
  }
  int status = memcmp(correct_results, gpu_results, THREADS * sizeof(int));

  if (status != 0){
    printf("FAIL\n");
    return 1;
  }
  printf("PASS\n");
  return 0;
}

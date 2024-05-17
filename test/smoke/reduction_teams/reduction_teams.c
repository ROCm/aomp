#include <stdio.h>
#include <omp.h>

#define N   1000000ll
#define SUM (N * (N-1)/2)

void checkHost(int gpu_error, int* errors, long long a){
  int host_error = 0;
  if (a != SUM){
    printf ("Host - Incorrect result = %lld, expected = %lld!\n", a, SUM);
    host_error = 1;
    (*errors)++;
  }
  if(!host_error && !gpu_error){
    printf("-----> Success\n");
  } else{
    printf("-----> Failure\n");
   }
}

void reduction(int num_teams, int num_threads, int* errors){
  long long result = 0;
  int gpu_error = 0;
  int device_teams = 0;
  int device_threads = 0;
  #pragma omp target teams num_teams(num_teams) thread_limit(num_threads) map(tofrom: result) map(from:device_teams,device_threads)
  {
    long long a, i;
    a = 0;
  #pragma omp parallel for reduction(+:a)
    for (i = 0; i < N; i++) {
        a += i;
	device_threads = omp_get_num_threads();
    }
    result = a;
      if (a != SUM && omp_get_team_num() <= 50){ //limit teams that print
        printf ("GPU - Incorrect result = %lld, expected = %lld!\n", a, SUM);
        gpu_error = 1;
      } 
    device_teams = omp_get_num_teams();
  } //end of target

  // Spec says you cannot have more teams than num_teams clause
  if ( device_teams > num_teams )  {
    (*errors)++;
    gpu_error++;
    printf("ERROR: num_teams requested:%d  actual teams on device:%d\n",
           num_teams, device_teams);
  }
  // Spec says you cannot have more threads than thread_limit clause
  if ( device_threads > num_threads ) {
    (*errors)++;
    gpu_error++;
    printf("ERROR:  num_threads limit:%d  Actual threads on device:%d\n",
           num_threads, device_threads);
  }
 
  checkHost(gpu_error, errors, result);
}


int main (void)
{
  int errors = 0;
  int gpu_error = 0;
 
  printf("\n---------- Multiple Teams ----------\n");
  printf("\nRunning 2 Teams with 64 thread per team\n");
  reduction(2, 64, &errors);

  printf("\nRunning 2 Teams with 128 threads per team\n");
  reduction(2, 128, &errors);

  printf("\nRunning 2 Teams with 256 threads per team\n");
  reduction(2, 256, &errors);
  
  printf("\nRunning 256 Teams with 256 threads per team (Limited to print first 50 teams)\n");
  reduction(256, 256, &errors);
 
  printf("\nRunning 4096 Teams with 64 threads per team (Limited to print first 50 teams)\n");
  reduction(4096, 64, &errors);
  
  printf("\nRunning 4096 Teams with 256 threads per team (Limited to print first 50 teams)\n");
  reduction(4096, 256, &errors);
  
  if(!errors){
    printf("\nRESULT: ALL RUNS SUCCESSFUL!\n");
    return 0;
  } else{
    printf("\nRESULT: FAILURES OCCURED!\n");
    return -1;
   }
}


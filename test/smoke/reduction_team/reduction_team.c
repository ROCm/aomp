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

void reduction(int num_teams, const int num_threads, int* errors){
  long long a = 0;
  int gpu_error = 0;
  #pragma omp target teams num_teams(num_teams) thread_limit(num_threads) map(tofrom: a)
  {
    long long i;

  #pragma omp parallel for reduction(+:a)
    for (i = 0; i < N; i++) {
        a += i;
    }
      if (a != SUM){
        printf ("GPU - Incorrect result = %lld, expected = %lld!\n", a, SUM);
        gpu_error = 1;
      } 
  }
  checkHost(gpu_error, errors, a);
}


void reduction_256(int num_teams, const int num_threads, int* errors){
  long long a = 0;
  int gpu_error = 0;
  #pragma omp target teams num_teams(num_teams) thread_limit(256) map(tofrom: a)
  {
    long long i;

  #pragma omp parallel for reduction(+:a)
    for (i = 0; i < N; i++) {
        a += i;
    }
      if (a != SUM){
        printf ("GPU - Incorrect result = %lld, expected = %lld!\n", a, SUM);
        gpu_error = 1;
      } 
  }
  checkHost(gpu_error, errors, a);
}

void reduction_512(int num_teams, const int num_threads, int* errors){
  long long a = 0;
  int gpu_error = 0;
  #pragma omp target teams num_teams(num_teams) thread_limit(512) map(tofrom: a)
  {
    long long i;

  #pragma omp parallel for reduction(+:a)
    for (i = 0; i < N; i++) {
        a += i;
    }
      if (a != SUM){
        printf ("GPU - Incorrect result = %lld, expected = %lld!\n", a, SUM);
        gpu_error = 1;
      } 
  }
  checkHost(gpu_error, errors, a);
}


void reduction_1024(int num_teams, const int num_threads, int* errors){
  long long a = 0;
  int gpu_error = 0;
  #pragma omp target teams num_teams(num_teams) thread_limit(1024) map(tofrom: a)
  {
    long long i;

  #pragma omp parallel for reduction(+:a)
    for (i = 0; i < N; i++) {
        a += i;
    }
      if (a != SUM){
        printf ("GPU - Incorrect result = %lld, expected = %lld!\n", a, SUM);
        gpu_error = 1;
      } 
  }
  checkHost(gpu_error, errors, a);
}

int main (void)
{
  int errors = 0;
  int gpu_error = 0;
  printf("\n---------- 1 Team with Variable Threads ----------\n");
  printf("\nRunning 1 Team with 64 threads per team\n");
  reduction(1, 64, &errors);
  
  printf("\nRunning 1 Team with 128 threads per team\n");
  reduction(1, 128, &errors);
 
 //Have to call a different function to use a constant for num_threads because
 //a variable will not allow the num_threads to go above 256 
  printf("\nRunning 1 Team with 256 threads per team\n");
  reduction_256(1, 256, &errors);
  
  printf("\nRunning 1 Team with 512 threads per team\n");
  reduction_512(1, 512, &errors);
  
  printf("\nRunning 1 Team with 1024 threads per team\n");
  reduction_1024(1, 1024, &errors);
 
  if(!errors){
    printf("\nRESULT: ALL RUNS SUCCESSFUL!\n");
    return 0;
  } else{
    printf("\nRESULT: FAILURES OCCURED!\n");
    return -1;
   }
}


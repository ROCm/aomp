#include <stdio.h>
#include <time.h>
#include <omp.h>

#define MAX_TEAMS 2048

#define TRIALS (1000)

int n =1024;

int main(void) {

  struct timespec t0,t1,t2;

  int fail = 0;
  int a = -1;
  //
  clock_gettime(CLOCK_REALTIME, &t0);
  #pragma omp target
  { //nothing
  }
  clock_gettime(CLOCK_REALTIME, &t1);
  double m = (t1.tv_sec - t0.tv_sec) + (t1.tv_nsec - t0.tv_nsec)/1e9;
  fprintf(stderr, "1st kernel Time %12.8f\n", m);
  for (int j = 1; j <= MAX_TEAMS; j = j<<1) {
    clock_gettime(CLOCK_REALTIME, &t1);
    for (int t = 0 ; t < TRIALS ; t++) {
      #pragma omp target teams distribute num_teams(j) thread_limit(1024)
      for (int k =0; k < n; k++)
      {
        // nothing
      }
    }
    clock_gettime(CLOCK_REALTIME, &t2);
    double t = (t2.tv_sec - t1.tv_sec) + (t2.tv_nsec - t1.tv_nsec)/1e9;
    fprintf(stderr, "avg kernel Time %12.8f TEAMS=%d\n", t/TRIALS, j);
  }
  printf("Succeeded\n");

  return fail;
}


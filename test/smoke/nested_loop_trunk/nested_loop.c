#include <stdio.h>
#include "assert.h"
#include <unistd.h>

#define TRIALS 1
#define N 960

int main() {
  int fail = 0;
  double A[N], B[N], C[N];
  for (int i = 0; i < N; i++) {
    A[i] = 0.0;
    B[i] = 0.0;
    C[i] = 1.0;
  }
  int nte = 32;
  int tl =  6;
  int blockSize =  5;
  int Inner=0, Outer=0;
  int wayout=0;

  for (int t = 0 ; t < TRIALS ; t++) {
    #pragma omp target map(tofrom: Inner, Outer, wayout, A)
    #pragma omp teams num_teams(1  ) thread_limit(tl)
    {   wayout =1;
//    #pragma omp distribute
  //  for(int j = 0 ; j < 16 ; j += blockSize) {
        #pragma omp parallel for
        for(int i = 0 ; i < blockSize; i++) {
          A[i] += B[i] + C[i];
	  Inner = 1;

        }
        Outer = 1;
  //  }
    }
  }
  printf("Inner=%d Outer=%d wayout=%d\n", Inner, Outer, wayout);
  for(int i = 0 ; i < 5 ; i++) {
    if (A[i] != TRIALS) {
      printf("Error at A[%d], h = %lf, d = %lf\n", i, (double)TRIALS, A[i]);
      fail = 1;
      break;
    }
  }

  if(fail){
	printf("Failed\n");
	return 1;
  }

  else{
	printf("Succeeded\n");
	return 0;
  }
}

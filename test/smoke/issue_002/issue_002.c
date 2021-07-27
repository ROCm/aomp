#include <stdio.h>
#include "assert.h"
#include <unistd.h>

// 920 fails
#define TRIALS 452
// 6000 fails
#define N 64*5000

int main() {
  int fail = 0;
  double A[N], B[N], C[N];
  for (int i = 0; i < N; i++) {
    A[i] = 0.0;
    B[i] = 0.0;
    C[i] = 1.0;
  }
  int nte = 32;
  int tl = 64;
  int blockSize = tl;

  for (int t = 0 ; t < TRIALS ; t++) {
    #pragma omp target
    #pragma omp teams num_teams(nte) thread_limit(tl)
    {
      #pragma omp distribute
      for(int j = 0 ; j < N ; j += blockSize) {
        #pragma omp parallel for
        for(int i = j ; i < j+blockSize; i++) {
          A[i] += B[i] + C[i];
        }
      }
    }
  }
  for(int i = 0 ; i < N ; i++) {
    if (A[i] != TRIALS) {
      printf("Error at A[%d], h = %lf, d = %lf\n", i, (double) (2.0+3.0)*TRIALS, A[i]);
      fail = 1;
      break;
    }
  }

  if(fail) {
	printf("Failed\n");
	return 1;
  }
  else {
	printf("Succeeded\n");
	return 0;
  }
}

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
  int tl = 64;
  int blockSize = tl;

  for (int t = 0 ; t < TRIALS ; t++) {
    #pragma omp target
    { 
      #pragma omp loop
      for(int j = 0 ; j < 256 ; j += blockSize) {
        for(int i = j ; i < j+blockSize; i++) {
          A[i] += B[i] + C[i];
        }
      }
    }
  }
  for(int i = 0 ; i < 256 ; i++) {
    if (A[i] != TRIALS) {
      printf("Error at A[%d], h = %lf, d = %lf\n", i, (double)TRIALS, A[i]);
      fail = 1;
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

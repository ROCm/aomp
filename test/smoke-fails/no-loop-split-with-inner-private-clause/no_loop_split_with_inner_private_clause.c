#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

const int N = 128*2;

int main(int argc, char **argv) {
float matA[N][N];
float matB[N][N];
float matC[N][N];
float matE[N][N];

  fprintf(stderr, "Starting matmul\n");
  #pragma omp target data map(to: matA, matB) map(from: matC) 
  {
    float tmp;
#pragma omp target map(to:N) //collapse(2)
#pragma omp teams distribute parallel for private(tmp)
    for (int i = 0; i < N; i++) {
      for (int j = 0; j < N; j++) {
        tmp = 0.0;
        for (int k = 0; k < N; k++) {
          tmp += matA[i][k] * matB[k][j];
        }
        matC[i][j] = tmp;
      }
    }
  }
  fprintf(stderr, "Passed\n");
  return 0;
}

/// CHECK: DEVID:[[S:[ ]*]][[DEVID:[0-9]+]] SGN:4

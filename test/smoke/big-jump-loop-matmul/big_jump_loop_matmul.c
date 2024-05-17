#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

const int N = 128*2;

void init(float A[][N], float B[][N]) {
  for (int i = 0; i < N; i++) {
    for (int k = 0; k < N; k++) {
      A[i][k] = i + k;
      B[i][k] = i + k + 1;
    }
  }
}

void reset(float C[][N]) {
  for (int i = 0; i < N; i++) {
    for (int k = 0; k < N; k++) {
      C[i][k] = 0;
    }
  }
}

int check(float matC[][N], float matCGolden[][N]) {
  for (int i = 0; i < N; i++) {
    for (int j = 0; j < N; j++) {
      if (matC[i][j] != matCGolden[i][j]) {
	fprintf(stderr, "Failed: matC[%d][%d] = %f matCGolden[%d][%d] = %f\n",
		i, j, matC[i][j], i, j, matCGolden[i][j]);
	return 1;
      }
    }
  }
  return 0;
}

int main(int argc, char **argv) {
  float matA[N][N];
  float matB[N][N];
  float matCGolden[N][N];
  float matC[N][N];

  init(matA, matB);
  
  fprintf(stderr, "Starting matmul on host\n");
  float tmp;
  for (int i = 0; i < N; i++) {
    for (int j = 0; j < N; j++) {
      tmp = 0.0;
      for (int k = 0; k < N; k++) {
	tmp += matA[i][k] * matB[k][j];
      }
      matCGolden[i][j] = tmp;
    }
  }
 
  fprintf(stderr, "Starting matmul on target, test 1\n");
  reset(matC);
  #pragma omp target data map(to: matA, matB) map(from: matC) 
  {
    float tmp;
#pragma omp target teams private(tmp) map(to:N) num_teams(20)
#pragma omp distribute parallel for collapse(2) 
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
  if (check(matC, matCGolden))
    return 1;

  fprintf(stderr, "Starting matmul on target, test 2\n");
  reset(matC);
  #pragma omp target data map(to: matA, matB) map(from: matC) 
  {
    float tmp;
#pragma omp target teams map(to:N) num_teams(20)
#pragma omp distribute parallel for collapse(2) private(tmp)
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
  if (check(matC, matCGolden))
    return 1;

  fprintf(stderr, "Starting matmul on target, test 3\n");
  reset(matC);
  #pragma omp target data map(to: matA, matB) map(from: matC) 
  {
    float tmp;
#pragma omp target map(to:N) private(tmp)
#pragma omp teams num_teams(20)
#pragma omp distribute parallel for collapse(2)
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
  if (check(matC, matCGolden))
    return 1;

  fprintf(stderr, "Starting matmul on target, test 4\n");
  reset(matC);
  #pragma omp target data map(to: matA, matB) map(from: matC) 
  {
    float tmp;
#pragma omp target map(to:N)
#pragma omp teams num_teams(20) private(tmp)
#pragma omp distribute parallel for collapse(2)
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
  if (check(matC, matCGolden))
    return 1;

  fprintf(stderr, "Starting matmul on target, test 5\n");
  reset(matC);
  #pragma omp target data map(to: matA, matB) map(from: matC) 
  {
    float tmp;
#pragma omp target map(to:N)
#pragma omp teams num_teams(20)
#pragma omp distribute parallel for collapse(2) private(tmp)
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
  if (check(matC, matCGolden))
    return 1;

  fprintf(stderr, "Starting matmul on target, test 6\n");
  reset(matC);
  #pragma omp target data map(to: matA, matB) map(from: matC) 
  {
    float tmp;
#pragma omp target map(to:N) private(tmp)
#pragma omp teams num_teams(20) private(tmp)
#pragma omp distribute parallel for collapse(2) private(tmp)
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
  if (check(matC, matCGolden))
    return 1;

  fprintf(stderr, "Starting matmul on target, test 7\n");
  reset(matC);
  #pragma omp target data map(to: matA, matB) map(from: matC) 
  {
    float tmp;
#pragma omp target map(to:N)
#pragma omp teams distribute parallel for collapse(2) num_teams(20) private(tmp)
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
  if (check(matC, matCGolden))
    return 1;

  fprintf(stderr, "Starting matmul on target, test 8\n");
  reset(matC);
  #pragma omp target data map(to: matA, matB) map(from: matC) 
  {
    float tmp;
#pragma omp target teams distribute parallel for private(tmp) map(to:N) collapse(2) num_teams(20)
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
  if (check(matC, matCGolden))
    return 1;

  fprintf(stderr, "Passed\n");
  return 0;
}

/// CHECK: DEVID:[[S:[ ]*]][[DEVID:[0-9]+]] SGN:5 ConstWGSize:256  args:11 teamsXthrds:(  20X 256)
/// CHECK: DEVID:[[S:[ ]*]][[DEVID:[0-9]+]] SGN:5 ConstWGSize:256  args:11 teamsXthrds:(  20X 256)
/// CHECK: DEVID:[[S:[ ]*]][[DEVID:[0-9]+]] SGN:5 ConstWGSize:256  args:11 teamsXthrds:(  20X 256)
/// CHECK: DEVID:[[S:[ ]*]][[DEVID:[0-9]+]] SGN:5 ConstWGSize:256  args:11 teamsXthrds:(  20X 256)
/// CHECK: DEVID:[[S:[ ]*]][[DEVID:[0-9]+]] SGN:5 ConstWGSize:256  args:11 teamsXthrds:(  20X 256)
/// CHECK: DEVID:[[S:[ ]*]][[DEVID:[0-9]+]] SGN:5 ConstWGSize:256  args:11 teamsXthrds:(  20X 256)
/// CHECK: DEVID:[[S:[ ]*]][[DEVID:[0-9]+]] SGN:5 ConstWGSize:256  args:11 teamsXthrds:(  20X 256)
/// CHECK: DEVID:[[S:[ ]*]][[DEVID:[0-9]+]] SGN:5 ConstWGSize:256  args:11 teamsXthrds:(  20X 256)

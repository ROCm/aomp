#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <cmath>

double get_seconds() {
  struct timespec now;
  clock_gettime(CLOCK_REALTIME, &now);

  const double seconds = (double) now.tv_sec;
  const double nsec    = (double) now.tv_nsec;

  return seconds + (nsec * 1.0e-9);
}

int main(int argc, char* argv[]) {
  // ------------------------------------------------------- //
  // DO NOT CHANGE CODE BELOW
  // ------------------------------------------------------- //

  int M = 256;
  int K = 256;
  int N = 256;
  int seed = 1;
  int reps = 1;

  double alpha = 1.0;
  double beta  = 1.0;

  double* matrixA = (double*) malloc(sizeof(double) * M * K);
  double* matrixB = (double*) malloc(sizeof(double) * K * N);
  double* matrixC = (double*) malloc(sizeof(double) * M * N);

  int i, j, k, r;
  srand48(seed);

  for(i = 0; i < M; i++)
    for(j = 0; j < K; j++)
      matrixA[i*K + j] = drand48()*2.0 - 1.0;

  for(i = 0; i < K; i++)
    for(j = 0; j < N; j++)
      matrixB[i*K + j] = drand48()*2.0 - 1.0;

  for(i = 0; i < M; i++)
    for(j = 0; j < N; j++)
      matrixC[i*K + j] = drand48()*2.0 - 1.0;

  const double start = get_seconds();

  for (r = 0; r < reps; r++) {

#pragma omp target teams distribute parallel for collapse(2) \
    map(to:matrixA[0: M*K]) \
    map(to:matrixB[0: K*N]) \
    map(tofrom:matrixC[0 : M*N])
    for(i = 0; i < M; i++) {
      for(j = 0; j < N; j++) {
        double sum = 0;

        for(k = 0; k < K; k++) {
          sum += matrixA[i*K + k] * matrixB[k*N + j];
        }

        matrixC[i*N + j] = (alpha * sum) + (beta * matrixC[i*N + j]);
      }
    }
  }

  const double end = get_seconds();

  double    final_sum = 0;
  long long int count     = 0;

  for(i = 0; i < M; i++) {
    for(j = 0; j < N; j++) {
      final_sum += matrixC[i*N + j];
      count++;
    }
  }

  double M_dbl = (double) M;
  double K_dbl = (double) K;
  double N_dbl = (double) N;
  double matrix_memory = (M_dbl * K_dbl) * ((double) sizeof(double))
                       + (K_dbl * N_dbl) * ((double) sizeof(double))
                       + (M_dbl * N_dbl) * ((double) sizeof(double));

  printf("\n");
  printf("===============================================================\n");

  const double count_dbl = (double) count;
  const double scaled_result = std::abs(final_sum) / count_dbl;

  printf("Final Sum is:         %f\n", scaled_result);

  printf("Memory for Matrices:  %f MB\n",
    (matrix_memory / (1024 * 1024)));

  const double time_taken = (end - start);
  const double reps_dbl   = (double)reps;

  printf("Input Size:           %d\n", M*K*N);
  printf("Total time:           %f seconds\n", time_taken);
  printf("Average time:         %f seconds\n", time_taken / reps_dbl);

  const double flops_computed = ((M_dbl * K_dbl * N_dbl * 2.0) +
          (M_dbl * N_dbl * 2.0))*reps;

  printf("FLOPs computed:       %f\n", flops_computed);
  printf("GFLOP/s rate:         %f GF/s\n", (flops_computed / time_taken) / 1000000000.0);

  printf("===============================================================\n");
  printf("\n");

  free(matrixA);
  free(matrixB);
  free(matrixC);

  return 0;
}

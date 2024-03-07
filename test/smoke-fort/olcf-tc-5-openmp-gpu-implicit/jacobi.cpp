#include <math.h>
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

const unsigned int n_cells = 512;
const unsigned int SIZE = (n_cells + 2) * (n_cells +2);

#define MAX(X, Y) ((X) > (Y) ? (X) : (Y))

#define T(i, j) (T[(i)*n_cells + (j)])
#define T_new(i, j) (T_new[(i)*n_cells + (j)])
#define T_results(i, j) (T_results[(i)*n_cells + (j)])

// smallest permitted change in temperature
double MAX_RESIDUAL = 1.e-5;

// Declaring T globally due to issue with nvhpc
static double T[SIZE];    // temperature grid
// initialize grid and boundary conditions
void init(double (&T)[SIZE]) {

  static int first_time = 1;
  static int seed = 0;
  if (first_time == 1) {
    seed = time(0);
    first_time = 0;
  }
  srand(seed);

  for (unsigned i = 0; i <= n_cells + 1; i++) {
    for (unsigned j = 0; j <= n_cells + 1; j++) {
      T(i, j) = (double)rand() / (double)RAND_MAX;
    }
  }
}

void kernel_serial(double (&T)[SIZE], int max_iterations) {

  int iteration = 0;
  double residual = 1.e5;
  double T_new[SIZE];

  // simulation iterations
  while (residual > MAX_RESIDUAL && iteration <= max_iterations) {

    // main computational kernel, average over neighbours in the grid
    for (unsigned i = 1; i <= n_cells; i++)
      for (unsigned j = 1; j <= n_cells; j++)
        T_new(i, j) =
            0.25 * (T(i + 1, j) + T(i - 1, j) + T(i, j + 1) + T(i, j - 1));

    // reset residual
    residual = 0.0;

    // compute the largest change and copy T_new to T
    for (unsigned int i = 1; i <= n_cells; i++) {
      for (unsigned int j = 1; j <= n_cells; j++) {
        residual = MAX(fabs(T_new(i, j) - T(i, j)), residual);
        T(i, j) = T_new(i, j);
      }
    }
    iteration++;
  }
  printf("Serial Residual = %.9lf\n", residual);

}

void kernel_gpu_teams_parallel_data_implicit(int max_iterations) {

  int iteration = 0;
  double residual = 1.e5;
  double T_new[SIZE];

  // simulation iterations
  while (residual > MAX_RESIDUAL && iteration <= max_iterations) {

    // main computational kernel, average over neighbours in the grid

#pragma omp target teams distribute parallel for simd collapse(2)
    for (unsigned i = 1; i <= n_cells; i++)
      for (unsigned j = 1; j <= n_cells; j++)
        T_new(i, j) =
            0.25 * (T(i + 1, j) + T(i - 1, j) + T(i, j + 1) + T(i, j - 1));

    // Reset residual
    residual = 0.0;

    // compute the largest change and copy T_new to T

#pragma omp target teams distribute parallel for simd collapse(2)  \
    reduction(max: residual) map(residual)
    for (unsigned int i = 1; i <= n_cells; i++) {
      for (unsigned int j = 1; j <= n_cells; j++) {
        residual = MAX(fabs(T_new(i, j) - T(i, j)), residual);
        T(i, j) = T_new(i, j);
      }
    }
    iteration++;
  }
  printf("Residual = %.9lf\n", residual);

}

void validate(double (&T)[SIZE], double (&T_results)[SIZE]) {

  double max_error = 0;
#pragma omp parallel for collapse(2) reduction(max : max_error)
  for (unsigned i = 1; i < n_cells; i++) {
    for (unsigned j = 1; j < n_cells; ++j) {
      double error = fabs(((T(i, j) - T_results(i, j)) / T_results(i, j)));
      max_error = MAX(error, max_error);
    }
  }

  printf("Validation maximum error = %.6lf : ", max_error);
  if (max_error < 20 * MAX_RESIDUAL) {
    printf("PASSED.\n");
  } else {
    printf(" VALIDATION ERROR.\n");
  }
}

int main(int argc, char *argv[]) {
  unsigned int max_iterations; // maximal number of iterations

  double T_results[SIZE]; // CPU results for validation

  if (argc < 2) {
    printf("Usage: %s number_of_iterations\n", argv[0]);
    exit(1);
  } else {
    max_iterations = atoi(argv[1]);
  }
  printf("Using statically sized number of cells = %u\n", n_cells);
  printf("Using maximum iteration count = %u\n", max_iterations);

  init(T_results);

  double start = omp_get_wtime();
  kernel_serial(T_results, max_iterations);
  double end = omp_get_wtime();

  double serial_cpu_time = end - start;

  init(T);

  start = omp_get_wtime();
  // Not passing T into the kernel due to issue with nvhpc
  kernel_gpu_teams_parallel_data_implicit(max_iterations);
  end = omp_get_wtime();

  validate(T, T_results);

  double omp_teams_parallel_data_time = end - start;

  printf("CPU serial kernel time = %.6lf Sec\n", serial_cpu_time);
  printf("OpenMP GPU Teams Distribute Parallel Implicit time = %.6lf Sec\n",
         omp_teams_parallel_data_time);
  printf("Number of OpenMP threads = %d\n", omp_get_max_threads());
  printf("Speedup = %.6lf\n", serial_cpu_time / omp_teams_parallel_data_time);
}

#include <math.h>
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

unsigned int n_cells;
unsigned int SIZE;

#define MAX(X, Y) ((X) > (Y) ? (X) : (Y))

#define T(i, j) (T[(i)*n_cells + (j)])
#define T_new(i, j) (T_new[(i)*n_cells + (j)])
#define T_results(i, j) (T_results[(i)*n_cells + (j)])

// smallest permitted change in temperature
double MAX_RESIDUAL = 1.e-5;

// initialize grid and boundary conditions
void init(double *T) {

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

void kernel_serial(double *T, int max_iterations) {

  int iteration = 0;
  double residual = 1.e5;
  double *T_new;

  T_new = (double *)malloc(SIZE * sizeof(double));

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

  free(T_new);
}

void kernel_omp_cpu(double *T, int max_iterations) {

  int iteration = 0;
  double residual = 1.e5;
  double *T_new;

  T_new = (double *)malloc((n_cells + 2) * (n_cells + 2) * sizeof(double));

  // simulation iterations
  while (residual > MAX_RESIDUAL && iteration <= max_iterations) {

    // main computational kernel, average over neighbours in the grid
#pragma omp parallel for collapse(2)
    for (unsigned i = 1; i <= n_cells; i++)
      for (unsigned j = 1; j <= n_cells; j++)
        T_new(i, j) =
            0.25 * (T(i + 1, j) + T(i - 1, j) + T(i, j + 1) + T(i, j - 1));

    // Reset residual
    residual = 0.0;

    // compute the largest change and copy T_new to T
#pragma omp parallel for reduction(max : residual) collapse(2)
    for (unsigned int i = 1; i <= n_cells; i++) {
      for (unsigned int j = 1; j <= n_cells; j++) {
        residual = MAX(fabs(T_new(i, j) - T(i, j)), residual);
        T(i, j) = T_new(i, j);
      }
    }
    iteration++;
  }
  printf("Residual = %.9lf\n", residual);

  free(T_new);
}

void kernel_gpu_teams(double *T, int max_iterations) {

  int iteration = 0;
  double residual = 1.e5;
  double *T_new;

  T_new = (double *)malloc((n_cells + 2) * (n_cells + 2) * sizeof(double));

  // simulation iterations
  while (residual > MAX_RESIDUAL && iteration <= max_iterations) {

    // main computational kernel, average over neighbours in the grid
#pragma omp target teams distribute collapse(2) map(T[:SIZE], T_new[:SIZE])
    for (unsigned i = 1; i <= n_cells; i++)
      for (unsigned j = 1; j <= n_cells; j++)
        T_new(i, j) =
            0.25 * (T(i + 1, j) + T(i - 1, j) + T(i, j + 1) + T(i, j - 1));

    // Reset residual
    residual = 0.0;

    // compute the largest change and copy T_new to T
#pragma omp target teams distribute collapse(2) reduction(max: residual)         \
    map(T[:SIZE], T_new[:SIZE]) 
    for (unsigned int i = 1; i <= n_cells; i++) {
      for (unsigned int j = 1; j <= n_cells; j++) {
        residual = MAX(fabs(T_new(i, j) - T(i, j)), residual);
        T(i, j) = T_new(i, j);
      }
    }
    iteration++;
  }
  printf("Residual = %.9lf\n", residual);

  free(T_new);
}

void kernel_gpu_teams_parallel(double *T, int max_iterations) {

  int iteration = 0;
  double residual = 1.e5;
  double *T_new;

  T_new = (double *)malloc((n_cells + 2) * (n_cells + 2) * sizeof(double));

  // simulation iterations
  while (residual > MAX_RESIDUAL && iteration <= max_iterations) {

    // main computational kernel, average over neighbours in the grid
#pragma omp target teams distribute parallel for simd collapse(2)              \
    map(T[:SIZE], T_new[:SIZE])
    for (unsigned i = 1; i <= n_cells; i++)
      for (unsigned j = 1; j <= n_cells; j++)
        T_new(i, j) =
            0.25 * (T(i + 1, j) + T(i - 1, j) + T(i, j + 1) + T(i, j - 1));

    // Reset residual
    residual = 0.0;

    // compute the largest change and copy T_new to T
#pragma omp target teams distribute parallel for simd collapse(2)              \
    reduction(max: residual) map(T[:SIZE], T_new[:SIZE]) 
    for (unsigned int i = 1; i <= n_cells; i++) {
      for (unsigned int j = 1; j <= n_cells; j++) {
        residual = MAX(fabs(T_new(i, j) - T(i, j)), residual);
        T(i, j) = T_new(i, j);
      }
    }
    iteration++;
  }
  printf("Residual = %.9lf\n", residual);

  free(T_new);
}

void kernel_gpu_teams_parallel_data(double *T, int max_iterations) {

  int iteration = 0;
  double residual = 1.e5;
  double *T_new;

  T_new = (double *)malloc((n_cells + 2) * (n_cells + 2) * sizeof(double));

#pragma omp target enter data map(to : T[:SIZE]) map(alloc : T_new[:SIZE])

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
    reduction(max: residual) 
    for (unsigned int i = 1; i <= n_cells; i++) {
      for (unsigned int j = 1; j <= n_cells; j++) {
        residual = MAX(fabs(T_new(i, j) - T(i, j)), residual);
        T(i, j) = T_new(i, j);
      }
    }
    iteration++;
  }
  printf("Residual = %.9lf\n", residual);

  free(T_new);
#pragma omp target update from(T[:SIZE])
#pragma omp target exit data map(delete : T[:SIZE]) map(delete : T_new[:SIZE])
}

void validate(double *T, double *T_results) {

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

  double *T;         // temperature grid
  double *T_results; // CPU results for validation

  if (argc < 3) {
    printf("Usage: %s number_of_cells number_of_iterations\n", argv[0]);
    exit(1);
  } else {
    n_cells = atoi(argv[1]);
    max_iterations = atoi(argv[2]);
  }
  printf("Using number of cells = %u\n", n_cells);
  printf("Using maximum iteration count = %u\n", max_iterations); 

  SIZE = (n_cells + 2) * (n_cells + 2);

  T = (double *)malloc(SIZE * sizeof(double));
  T_results = (double *)malloc(SIZE * sizeof(double));

  if (T == NULL || T_results == NULL) {
    printf("Error allocating storage for Temperature\n");
    exit(1);
  }

  init(T_results);
  double serial_kernel_time = 0.0;

  {
    init(T_results);
    printf("================ Serial Kernel ================================== \n");
    double start = omp_get_wtime();
    kernel_serial(T_results, max_iterations);
    double end = omp_get_wtime();
    serial_kernel_time = end - start;
    printf("CPU serial kernel time = %.6f Seconds\n", end - start);
    printf("================================================================= \n\n");
  }

  {
    init(T);
    double start = omp_get_wtime();
    printf("=============== CPU OpenMP Kernel =============================== \n");
    kernel_omp_cpu(T, max_iterations);
    double end = omp_get_wtime();
    validate(T, T_results);
    double omp_cpu_time = end - start;
    printf("OpenMP CPU time = %.6lf Sec\n", omp_cpu_time);
    printf("Number of OpenMP threads = %d\n", omp_get_max_threads());
    printf("Speedup = %.6lf\n", serial_kernel_time / omp_cpu_time);
    printf("================================================================= \n\n");
  }

  {
    init(T);
    printf("=============== GPU Teams Kernel =============================== \n");
    double start = omp_get_wtime();
    kernel_gpu_teams(T, max_iterations);
    double end = omp_get_wtime();
    validate(T, T_results);
    double omp_gpu_teams_time = end - start;
    printf("OpenMP GPU Teams time = %.6lf Sec\n", omp_gpu_teams_time);
    printf("Speedup = %.6lf\n", serial_kernel_time / omp_gpu_teams_time);
    printf("================================================================= \n\n");
  }

  {
    init(T);
    printf("=============== GPU Teams Distribute Parallel =================== \n");
    double start = omp_get_wtime();
    kernel_gpu_teams_parallel(T, max_iterations);
    double end = omp_get_wtime();
    validate(T, T_results);
    double omp_gpu_teams_parallel_time = end - start;
    printf("OpenMP GPU Teams Distribute Parallel time  = %.6lf Sec\n",
           omp_gpu_teams_parallel_time);
    printf("Speedup = %.6lf\n",
           serial_kernel_time / omp_gpu_teams_parallel_time);
    printf("================================================================= \n\n");
  }

  {
    printf("=============== GPU Teams Distribute Parallel Data =============== \n");
    init(T);
    double start = omp_get_wtime();
    kernel_gpu_teams_parallel_data(T, max_iterations);
    double end = omp_get_wtime();
    validate(T, T_results);
    double omp_gpu_teams_parallel_data_time = end - start;
    printf("OpenMP GPU Teams Distribute Parallel data time  = %.6lf Sec\n",
           omp_gpu_teams_parallel_data_time);
    printf("Speedup = %.6lf\n",
           serial_kernel_time / omp_gpu_teams_parallel_data_time);
    printf("================================================================= \n\n");
  }

  return 0;
}

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <omp.h>

// The program includes host, AMDGCN, and NVPTX implementations of SAXPY
// and uses OpenMP to select the appropriate variant based on the target device.

// --- Start of SAXPY header with variants ---
int saxpy(int, float, float *, float *);
int amdgcn_saxpy(int, float, float *, float *);
int nvptx_saxpy(int, float, float *, float *);

// Declare architecture-specific variants for SAXPY
#pragma omp declare variant(nvptx_saxpy) \
    match(device = {arch(nvptx, nvptx64)}, implementation = {extension(match_any)})
#pragma omp declare variant(amdgcn_saxpy) \
    match(device = {arch(amdgcn)}, implementation = {extension(match_any)})

// Base SAXPY function for host execution
int saxpy(int n, float s, float *x, float *y) {
  printf("saxpy: Running on host. IsHost:%d\n", omp_is_initial_device());
#pragma omp parallel for
  for (int i = 0; i < n; i++) y[i] = s * x[i] + y[i]; // Perform SAXPY operation
  return 1;
}

// Variant for AMDGCN devices
int amdgcn_saxpy(int n, float s, float *x, float *y) {
  printf("amdgcn_saxpy: Running on amdgcn device. IsHost:%d\n", omp_is_initial_device());
#pragma omp teams distribute parallel for
  for (int i = 0; i < n; i++) {
    y[i] = s * x[i] + y[i]; // Perform SAXPY operation
  }
  return 0;
}

// Variant for NVPTX devices
int nvptx_saxpy(int n, float s, float *x, float *y) {
  printf("nvptx_saxpy: Running on nvptx device. IsHost:%d\n", omp_is_initial_device());
#pragma omp teams distribute parallel for
  for (int i = 0; i < n; i++) y[i] = s * x[i] + y[i]; // Perform SAXPY operation
  return 0;
}

// --- End of SAXPY header with variants ---

#define N 128
#define THRESHOLD 127

int main() {
  // Allocate and align arrays
  static float x[N], y[N] __attribute__((aligned(64)));
  float s = 2.0;
  int return_code = 0;

  // Initialize arrays
  for (int i = 0; i < N; i++) {
    x[i] = i + 1;
    y[i] = i + 1;
  }

  // Call SAXPY with a high threshold for device execution
  printf("Calling saxpy with high threshold for device execution\n");
#pragma omp target if (N > (THRESHOLD * 2))
  return_code = saxpy(N, s, x, y);

  // Call SAXPY with a low threshold for device execution
  printf("Calling saxpy with low threshold for device execution\n");
#pragma omp target if (N > THRESHOLD)
  return_code = saxpy(N, s, x, y);

  // Print results
  printf("y[0],y[N-1]: %5.0f %5.0f\n", y[0], y[N - 1]); // Output: y[0] and y[N-1]

  return return_code;
}

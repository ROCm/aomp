// Minimal host: triggers rocBLAS Tensile lazy-load on first meaningful kernel.
// Build once; test different ROCm drops via LD_LIBRARY_PATH without rebuilding VASP.
#include <hip/hip_runtime.h>
#include <rocblas/rocblas.h>
#include <cstdio>
#include <cstdlib>

static void die_hip(hipError_t e, const char *what) {
  if (e != hipSuccess) {
    fprintf(stderr, "%s: %s\n", what, hipGetErrorString(e));
    std::exit(1);
  }
}

static void die_rb(rocblas_status s, const char *what) {
  if (s != rocblas_status_success) {
    fprintf(stderr, "%s: rocblas_status %d\n", what, static_cast<int>(s));
    std::exit(1);
  }
}

int main() {
  int dev = 0;
  die_hip(hipGetDevice(&dev), "hipGetDevice");
  hipDeviceProp_t prop{};
  die_hip(hipGetDeviceProperties(&prop, dev), "hipGetDeviceProperties");
  std::printf("HIP device %d: %s gcnArchName=%s\n", dev, prop.name, prop.gcnArchName);

  rocblas_handle handle{};
  die_rb(rocblas_create_handle(&handle), "rocblas_create_handle");

  const rocblas_int m = 256, n = 256, k = 256;
  const float alpha = 1.0f, beta = 0.0f;
  float *dA = nullptr, *dB = nullptr, *dC = nullptr;
  die_hip(hipMalloc(&dA, sizeof(float) * m * k), "hipMalloc A");
  die_hip(hipMalloc(&dB, sizeof(float) * k * n), "hipMalloc B");
  die_hip(hipMalloc(&dC, sizeof(float) * m * n), "hipMalloc C");
  die_hip(hipMemset(dA, 0, sizeof(float) * m * k), "hipMemset A");
  die_hip(hipMemset(dB, 0, sizeof(float) * k * n), "hipMemset B");
  die_hip(hipMemset(dC, 0, sizeof(float) * m * n), "hipMemset C");

  die_rb(rocblas_sgemm(handle, rocblas_operation_none, rocblas_operation_none, m, n, k,
                       &alpha, dA, m, dB, k, &beta, dC, m),
        "rocblas_sgemm");

  die_hip(hipDeviceSynchronize(), "hipDeviceSynchronize");
  std::puts("rocblas_sgemm completed (Tensile path exercised if applicable).");

  rocblas_destroy_handle(handle);
  (void)hipFree(dA);
  (void)hipFree(dB);
  (void)hipFree(dC);
  return 0;
}

#include <hip/hip_runtime.h>
#include <stdio.h>

// Testing hip compilation (-x hip) with -fopenmp. This should not pick up the openmp_wrappers/hip_host_overlay.h header. If it does then there will be an error:
// ld.lld: error: undefined symbol: omp_register_coarse_grain_mem
int main() {
  constexpr int num_objects = 2;
  void** buffers{nullptr};
  hipError_t mallocResult = hipHostMalloc(&buffers, num_objects * sizeof(void*));
  hipError_t freeResult = hipHostFree(buffers);
  if (mallocResult != hipSuccess || freeResult != hipSuccess){
    printf("Test failed in hipHostMalloc or hipHostFree \n");
    return 1;
  }

  mallocResult = hipMalloc(&buffers, num_objects * sizeof(void*));
  freeResult = hipFree(buffers);
  if (mallocResult != hipSuccess || freeResult != hipSuccess){
    printf("Test failed in hipMalloc or hipFree! \n");
    return 1;
  }
  printf("Passed\n");
  return 0;
}

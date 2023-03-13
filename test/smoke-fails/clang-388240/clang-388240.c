/* Simple OMP offload test */
/* Based on clang-387196 */

#include <omp.h>
#include <stdio.h>

int main(int argc, char* argv[]) {

int runningOnGPU = 0;

  /* Test if GPU is available using OpenMP4.5 */
#pragma omp target map(from:runningOnGPU)
  {
    if (omp_is_initial_device() == 0)
      runningOnGPU = 1;
  }
  /* If still running on CPU, GPU must not be available */
  if (runningOnGPU) {
    printf("PASS\n");
  } else {
    printf("FAIL - not running on a GPU\n");
    return 1;
  }

  return 0;
}


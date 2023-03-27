/*
Compile command:

amdclang -fopenmp -target x86_64-pc-linux-gnu -fopenmp-targets=amdgcn-amd-amdhsa
-Xopenmp-target=amdgcn-amd-amdhsa -march=gfx90a ./test_offload.c

*/

// CHECK: DEVID

#include <math.h>
#include <omp.h>
#include <stdio.h>

int main()

{
  int num_devices = omp_get_num_devices();
  printf("Number of available devices %d\n", num_devices);

#pragma omp target
  {
    printf("omp_is_initial_device() -> %d  omp_get_initial_device() -> %d "
           "(expected return values: 0 and -1)\n",
           omp_is_initial_device(), omp_get_initial_device());
    if (omp_is_initial_device()) {
      printf("Running on host\n");
    } else {
      int nteams = omp_get_num_teams();
      int nthreads = omp_get_num_threads();
      printf("Running on device with %d teams in total and %d threads in each "
             "team\n",
             nteams, nthreads);
    }
  }
}

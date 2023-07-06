#include <stdio.h>
#include <omp.h>
#include <chrono>
#include "hip_hostreg.h"

// 4KB of double's
#define ARR_SIZE 4096/8

#pragma omp requires unified_shared_memory

extern "C" {
void *llvm_omp_target_lock_mem(void *ptr, size_t size, int device_num);
}

int main(int argc, char **argv) {
  int arr_size = ARR_SIZE;
  if (argc > 1) arr_size = atoi(argv[1]);

  double *array = new double[arr_size]; 
  //  llvm_omp_target_lock_mem(array, arr_size*sizeof(double), 0);
  hip_hostreg(array, arr_size*sizeof(double));

  #pragma omp target teams distribute parallel for map(tofrom:array[:arr_size])
  for(int i = 0; i < arr_size; i++)
    array[i] = (double)i;

  int err = 0;
  for(int i = 0; i < arr_size; i++)
    if (array[i] != (double)i) {
      printf("Err at %d: got %lf, expected %lf\n", i, array[i], (double)i);
      err++;
      if (err > 10)
        return err;
    }

  return 0;
}

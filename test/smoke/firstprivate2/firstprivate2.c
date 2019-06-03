#include <stdio.h>
#include "assert.h"
#include <unistd.h>
#include <omp.h>

#define N 2
#define NUM_THREADS 3

int main(){
  int array1[NUM_THREADS];
  int array2[N*NUM_THREADS];
  //omp_set_num_threads(NUM_THREADS);
  for (int i = 0; i < NUM_THREADS; i++){
    array1[i] = -1;
  }
  for (int i = 0; i < N * NUM_THREADS; i++){
    array2[i] = -1;
  }

#pragma omp parallel num_threads(3)
{
  int p_val = omp_get_thread_num();
  fprintf(stderr,"Thread num: %d P_VAL: %d\n", omp_get_thread_num(),p_val);

#pragma omp target firstprivate(p_val)
{
  array1[p_val] = p_val + 100;
  for(int x = 0; x < N; x++)
    array2[p_val * N + x] = 200;
}
}
  //Print Arrays
  for(int i = 0; i < NUM_THREADS; i++){
    fprintf(stderr, "Array1[%d]: %d\n", i, array1[i]);
  }
  fprintf(stderr, "\n");

  for(int i = 0; i < N*NUM_THREADS; i++){
    fprintf(stderr, "Array2[%d]: %d\n", i, array2[i]);
  }

  return 0;
}



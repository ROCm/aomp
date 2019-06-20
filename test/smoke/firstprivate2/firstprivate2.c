#include <stdio.h>
#include "assert.h"
#include <unistd.h>
#include <omp.h>

#define N 2
#define NUM_THREADS 3

int main(){
  int array1[NUM_THREADS];
  int array2[N*NUM_THREADS];
  int errors = 0;

  //omp_set_num_threads(NUM_THREADS);
  for (int i = 0; i < NUM_THREADS; i++){
    array1[i] = -1;
  }
  for (int i = 0; i < N * NUM_THREADS; i++){
    array2[i] = -1;
  }
#pragma omp target data map(from:array1, array2)
#pragma omp parallel num_threads(3)
{
  int p_val = omp_get_thread_num();
  fprintf(stderr,"Thread num: %d P_VAL: %d\n", omp_get_thread_num(),p_val);

#pragma omp target firstprivate(p_val)
{
  array1[p_val] = p_val + 100;
  for(int x = 0; x < N; x++)
    array2[p_val * N + x] = 200;
  p_val++;
}
  if(p_val != omp_get_thread_num()){
    printf("Unwanted Behavior: P_VAL Changed to: %d. Should be %d.\n", p_val, omp_get_thread_num());
    errors = 1;
  }
}
  //Print Arrays
  for(int i = 0; i < NUM_THREADS; i++){
    fprintf(stderr, "Array1[%d]: %d\n", i, array1[i]);
    if(array1[i] != 100 + i){
      printf("Array1 has invalid value %d  at index %d.\n", array1[i], i);
      errors = 1;
    }
  }
  fprintf(stderr, "\n");

  for(int i = 0; i < N*NUM_THREADS; i++){
    fprintf(stderr, "Array2[%d]: %d\n", i, array2[i]);
    if(array2[i] != 200){
      printf("Array2 has invalid value %d  at %d.\n", array2[i], i);
      errors = 1;
    }
  }
  if(errors){
    printf("Fail!\n");
    return 1;
  }
  printf("Success!\n");
  return 0;
}



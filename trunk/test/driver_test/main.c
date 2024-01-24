#include <stdio.h>

#define N 10

#include "_myomplib.h"

void writeIndex(int *b, int n) {
#pragma omp target teams distribute parallel for map(tofrom: b[0:n])
  for (int i = 0; i < n; ++i) {
    b[i] = i;
    inc_arrayval(i, b ) ; // c
    dec_arrayval(i, b ) ; // FORTRAN
  }
}

void printArray(int *array) {
  printf("[");
  int first = 1 ;
  for (int i = 0; i < N; ++i) {
    if (first) {
      printf("%d", array[i]);
      first = 0;
    } else {
      printf(", %d", array[i]);
    }
  }
  printf("]");
}

int checkArray(int *array){
  int errors = 0;
  for(int i = 0; i < N; ++i){
    if(array[i] != i)
      errors++;
  }
  return errors;
}

int main() {
  int hostArray[N];

  for (int i = 0; i < N; ++i)
    hostArray[i] = 0;

  printf("Array content before target:\n");
  printArray(hostArray);
  printf("\n");

  writeIndex(hostArray,N);

  printf("Array content after target:\n");
  printArray(hostArray);
  printf("\n");

  int errors = checkArray(hostArray);

  if(errors){
    printf("Fail!\n");
    return 1;
  }
  printf("Success!\n");
  return 0;
}

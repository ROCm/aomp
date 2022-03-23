#include <stdio.h>

int main (void) {
  int sum = 0;
  #pragma omp parallel for reduction(+:sum)
  for(int i = 0 ; i < 20000000; i++) {
    sum += i%7;
  }
  printf("CPU sum = %d\n",sum);
  sum=0;
  #pragma omp target teams distribute parallel for reduction(+:sum)
  for(int i = 0 ; i < 20000000; i++) {
    sum += i%7;
  }
  printf("GPU sum = %d\n",sum);
  return 0;
}

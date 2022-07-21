#include <stdio.h>

int main (void) {
  int sum = 0;
  int gsum = 0;
  #pragma omp parallel for reduction(+:sum)
  for(int i = 0 ; i < 20000000; i++) {
    sum += i%7;
  }
  printf("CPU sum = %d\n",sum);
  gsum=0;
  #pragma omp target teams distribute parallel for reduction(+:gsum)
  for(int i = 0 ; i < 20000000; i++) {
    gsum += i%7;
  }
  printf("GPU sum = %d\n",gsum);
  if (gsum != sum) return 1;
  return 0;
}

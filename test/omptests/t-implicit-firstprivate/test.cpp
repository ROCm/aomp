#include <omp.h>
#include <stdio.h>
#include <math.h>

int A = 10;

int main(void) {

  int B = 20;
  static int C = 30;

  #pragma omp target map(A, B, C)
  {
    #pragma omp parallel num_threads(1)
    {
      ++A; ++B; ++C;
    }
  }
  
  printf("A -> %d, B -> %d, C -> %d\n", A, B, C);
  
  return 0;
}

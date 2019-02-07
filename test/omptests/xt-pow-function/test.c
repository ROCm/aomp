
#include <stdio.h>
#include <omp.h>
#include <math.h>

int main(void) {
  double A = 2.0;
  float B = 2.0;

  #pragma omp target map(A,B)
  {
    A = powi(A, 4);
    //B = powif(B, 4);
  }

  printf("%lf\n",A);
  //printf("%f\n",B);
  return 0;
}

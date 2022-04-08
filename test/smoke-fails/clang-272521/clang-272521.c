#include <stdio.h>
#include <complex.h>

int main() {
  int i;
  float complex C = 0 + 0 * I;
#pragma omp target parallel for reduction(+:C) map(tofrom: C)
  for (int i = 0; i <10; i++) {
     C += 1.0 + 1.0*I;
  }
  printf("C = %f+%fi\n", creal(C),cimag(C));
  return 0;
}

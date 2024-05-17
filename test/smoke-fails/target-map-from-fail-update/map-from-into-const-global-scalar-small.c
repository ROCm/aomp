#include "stdio.h"

const int N = 128;
// When const is removed, the test works
const int A[128] = {0};
int M = 1;

int main(int argc, char **argv) {
  int tmp = 0;

#pragma omp target map(to: N) map(tofrom: tmp)
  {
    tmp = N + 1;
  }

  if ((tmp - (N + 1))) {
    return -1;
  }

#pragma omp target map(tofrom: N) map(from:tmp)
  {
    tmp = N + 2;
  }
  
  if ((tmp - (N + 2))) {
    return -2;
  }

  fprintf(stderr, "Passed\n");

  return 0;
}

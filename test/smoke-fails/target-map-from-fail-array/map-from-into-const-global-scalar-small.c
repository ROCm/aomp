#include "stdio.h"

const int N = 128;
// When const is removed, the test works
const int A[128] = {0};
int M = 1;

int main(int argc, char **argv) {
  int tmp = 0;
  int tmp2 = 0;

  for (int i =0; i < N; ++i) {
    tmp2 += A[i] + 1;
  }

#pragma omp target teams distribute parallel for reduction(+:tmp) map(A, N, M) map(tofrom:M)
  for (int i = 0; i < N; i++) {
    tmp += A[i] + 1;
    M = 42;
  }

  fprintf(stderr, "tmp: %u == %u\n", tmp, tmp2);
  fprintf(stderr, "M: %u == 42\n", M);


  fprintf(stderr, "Passed\n");

  return tmp-tmp2;
}

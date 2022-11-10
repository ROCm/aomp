#include "omp.h"
#include "stdio.h"

const int N = 128;
int M = 256;

int main(int argc, char **argv) {
const int K = 64;
  int tmp = 0;
#pragma omp target teams distribute parallel for private(tmp) map(to:N) map(M) map(to:K)
  for (int i = 0; i < N; i++) {
    tmp += i;
  }

  fprintf(stderr, "Passed\n");

  return 0;
}

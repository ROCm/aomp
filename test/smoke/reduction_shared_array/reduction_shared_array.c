#include <omp.h>
#include <stdio.h>

#define ITERS 4096

int main() {
  #pragma omp target
  {
    int globalized[256];

    #pragma omp parallel for
    for (int i = 0; i < 256; i++) {
      globalized[i] = 0;
    }
    #pragma omp parallel for reduction(+:globalized)
    for (int i = 0; i < ITERS; i++) {
      globalized[i % 256] += i;
    }

    printf("%d", globalized[0]);
    for (int i = 1; i < 256; i++) {
      printf(" %d", globalized[i]);
    }
    printf("\n");
  }

  return 0;
}

#include <stdlib.h>
#include <stdio.h>

#define N 10000

int main() {
  int n = N;
  int *a = (int *)malloc(n*sizeof(int));

  #pragma omp teams distribute
  for(int i = 0; i < n; i++)
    a[i] = i;

  int err = 0;
  for(int i = 0; i < n; i++)
    if (a[i] != i) {
      printf("Error at %d: a = %d, should be %d\n", i, a[i], i);
      err++;
      if (err > 10) break;
    }

  return err;
}

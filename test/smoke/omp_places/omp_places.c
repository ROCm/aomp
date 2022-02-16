#include <omp.h>

#define N 1000000000
int main() {
  long int n = N;
  int *a = (int *)malloc(n*sizeof(int));

  #pragma omp teams distribute parallel for
  {
        for (long int i = 0; i < n; i++) {
          a[i] = i;
        }
  }
  free(a);
  return 0;
}

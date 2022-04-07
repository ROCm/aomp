#include <stdlib.h>
#include <stdio.h>
#include <omp.h>
#include <sys/time.h>

#define N 1000000000
int main() {
  long int n = N;
  int *a = (int *)malloc(n*sizeof(int));
  int err = 0;
  struct timeval start, stop;
  double elapsed;

  for (long int i = 0; i < n; i++)
      a[i] = 1;
  
  gettimeofday(&start, NULL);
  #pragma omp teams distribute parallel for
  {
        for (long int i = 0; i < n; i++) {
          a[i] = i;
        }
  }
  gettimeofday(&stop, NULL);
  elapsed = (stop.tv_sec - start.tv_sec) + ((stop.tv_usec - start.tv_usec) * 1e-6);
  printf("Total time for host teams directive : %g\n", elapsed);
  err = 0;
  for (long int i = 0; i < n; i++) {
          if (a[i] != i) {
              printf("Error at %ld: a = %d, should be %ld\n", i, a[i], i);
              err++;
              if (err > 10) break;
          }
  }
  free(a);
  return err;
}

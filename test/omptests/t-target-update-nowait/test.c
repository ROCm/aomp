#include <stdio.h>
#define N 1024

#define TEST_UPDATE     1
#define TEST_COMPUTE_UPDATE 1

int a[N], b[N];

int main() {
  int i;
  int error, totError = 0;


#if TEST_UPDATE
  for (i=0; i<N; i++) a[i] = b[i] = i;
  #pragma omp target data map(to:b) map(alloc: a)
  {

      #pragma omp target map(to:b) map(alloc: a)
      {
        int j;
        for(j=0; j<N/4; j++) a[j] = b[j]+1;
      }
      #pragma omp target update from(a[0:N/4]) nowait

      #pragma omp target  map(to:b) map(alloc: a)
      {
        int j;
        for(j=N/4; j<N/2; j++) a[j] = b[j]+1;
      }
      #pragma omp target update from(a[N/4:N/4]) nowait

      #pragma omp target map(to:b) map(alloc: a)
      {
        int j;
        for(j=N/2; j<3*(N/4); j++) a[j] = b[j]+1;
      }
      #pragma omp target update from(a[N/2:N/4]) nowait

      #pragma omp target map(to:b) map(alloc: a)
      {
        int j;
        for(j=3*(N/4); j<N; j++) a[j] = b[j]+1;
      }
      #pragma omp target update from(a[3*(N/4):N/4]) nowait

      #pragma omp taskwait
  }

  error=0;
  for (i=0; i<N; i++) {
    if (a[i] != i+1) printf("%d: error %d != %d, error %d\n", i, a[i], i+1, ++error);
  }
  if (! error) {
    printf("  test with update conpleted successfully\n");
  } else {
    printf("  test with update conpleted with %d error(s)\n", error);
    totError++;
  }
#endif

#if TEST_COMPUTE_UPDATE
  for (i=0; i<N; i++) a[i] = b[i] = i;
  #pragma omp target data map(to:b) map(alloc: a)
  {

      #pragma omp target nowait map(to:b) map(alloc: a)
      {
        int j;
        for(j=0; j<N/4; j++) a[j] = b[j]+1;
      }

      #pragma omp target nowait map(to:b) map(alloc: a)
      {
        int j;
        for(j=N/4; j<N/2; j++) a[j] = b[j]+1;
      }

      #pragma omp taskwait
      //printf("waited, initiate first update from\n");
      #pragma omp target update from(a[0:N/4]) nowait
      #pragma omp target update from(a[N/4:N/4]) nowait

      #pragma omp target nowait map(to:b) map(alloc: a)
      {
        int j;
        for(j=N/2; j<3*(N/4); j++) a[j] = b[j]+1;
      }

      #pragma omp target nowait map(to:b) map(alloc: a)
      {
        int j;
        for(j=3*(N/4); j<N; j++) a[j] = b[j]+1;
      }

      #pragma omp taskwait
      //printf("waited, initiate second update from\n");

      #pragma omp target update from(a[N/2:N/4]) nowait
      #pragma omp target update from(a[3*(N/4):N/4]) nowait

      #pragma omp taskwait
  }

  error=0;
  for (i=0; i<N; i++) {
    if (a[i] != i+1) printf("%d: error %d != %d, error %d\n", i, a[i], i+1, ++error);
  }
  if (! error) {
    printf("  test with compute update conpleted successfully\n");
  } else {
    printf("  test with compute update conpleted with %d error(s)\n", error);
    totError++;
  }
#endif


  printf("completed with %d errors\n", totError);
  return totError;
}

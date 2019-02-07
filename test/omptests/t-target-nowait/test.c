#include <stdio.h>
#define N 1024

#define TEST_NESTED     1
#define TEST_CONCURRENT 1
#define TEST_CONCURRENT_TF 1
#define TEST_PARALLEL1     1

int a[N], b[N];

int main() {
  int i;
  int error, totError = 0;


#if TEST_NESTED
  for (i=0; i<N; i++) a[i] = b[i] = i;
  #pragma omp target data map(to:b) map(from: a)
  {

      #pragma omp target nowait map(to:b) map(from: a)
      {
        int j;
        for(j=0; j<N/4; j++) a[j] = b[j]+1;
      }

      #pragma omp target nowait map(to:b) map(from: a)
      {
        int j;
        for(j=N/4; j<N/2; j++) a[j] = b[j]+1;
      }

      #pragma omp target nowait map(to:b) map(from: a)
      {
        int j;
        for(j=N/2; j<3*(N/4); j++) a[j] = b[j]+1;
      }

      #pragma omp target nowait map(to:b) map(from: a)
      {
        int j;
        for(j=3*(N/4); j<N; j++) a[j] = b[j]+1;
      }

      #pragma omp taskwait
  }

  error=0;
  for (i=0; i<N; i++) {
    if (a[i] != i+1) printf("%d: error %d != %d, error %d\n", i, a[i], i+1, ++error);
  }
  if (! error) {
    printf("  test with nested maps conpleted successfully\n");
  } else {
    printf("  test with nested maps conpleted with %d error(s)\n", error);
    totError++;
  }
#endif

#if TEST_CONCURRENT_TF
  for (i=0; i<N; i++) a[i] = b[i] = i;

  #pragma omp target nowait map(tofrom:a, b) 
  {
    int j;
    for(j=0; j<N/4; j++) a[j] = b[j]+1;
  }

  #pragma omp target nowait map(tofrom:a, b) 
  {
    int j;
    for(j=N/4; j<N/2; j++) a[j] = b[j]+1;
  }

  #pragma omp target nowait map(tofrom:a, b) 
  {
    int j;
    for(j=N/2; j<3*(N/4); j++) a[j] = b[j]+1;
  }

  #pragma omp target nowait map(tofrom:a, b) 
  {
    int j;
    for(j=3*(N/4); j<N; j++) a[j] = b[j]+1;
  }

  #pragma omp taskwait

  error=0;
  for (i=0; i<N; i++) {
    if (a[i] != i+1) printf("%d: error %d != %d, error %d\n", i, a[i], i+1, ++error);
  }
  if (! error) {
    printf("  test with concurrent with to/from maps conpleted successfully\n");
  } else {
    printf("  test with concurrent with to/from maps conpleted with %d error(s)\n", error);
    totError++;
  }
#endif


#if TEST_CONCURRENT
  for (i=0; i<N; i++) a[i] = b[i] = i;

  #pragma omp target nowait map(to:b) map(from: a)
  {
    int j;
    for(j=0; j<N/4; j++) a[j] = b[j]+1;
  }

  #pragma omp target nowait map(to:b) map(from: a)
  {
    int j;
    for(j=N/4; j<N/2; j++) a[j] = b[j]+1;
  }

  #pragma omp target nowait map(to:b) map(from: a)
  {
    int j;
    for(j=N/2; j<3*(N/4); j++) a[j] = b[j]+1;
  }

  #pragma omp target nowait map(to:b) map(from: a)
  {
    int j;
    for(j=3*(N/4); j<N; j++) a[j] = b[j]+1;
  }

  #pragma omp taskwait

  error=0;
  for (i=0; i<N; i++) {
    if (a[i] != i+1) printf("%d: error %d != %d, error %d\n", i, a[i], i+1, ++error);
  }
  if (! error) {
    printf("  test with concurrent maps conpleted successfully\n");
  } else {
    printf("  test with concurrent maps conpleted with %d error(s)\n", error);
    totError++;
  }
#endif

#if TEST_PARALLEL1
  for (i=0; i<N; i++) a[i] = b[i] = i;
  #pragma omp parallel num_threads(1)
  {
    #pragma omp target data map(to:b) map(from: a)
    {

      #pragma omp target nowait map(to:b) map(from: a)
      {
        int j;
        for(j=0; j<N/4; j++) a[j] = b[j]+1;
      }

      #pragma omp target nowait map(to:b) map(from: a)
      {
        int j;
        for(j=N/4; j<N/2; j++) a[j] = b[j]+1;
      }

      #pragma omp target nowait map(to:b) map(from: a)
      {
        int j;
        for(j=N/2; j<3*(N/4); j++) a[j] = b[j]+1;
      }

      #pragma omp target nowait map(to:b) map(from: a)
      {
        int j;
        for(j=3*(N/4); j<N; j++) a[j] = b[j]+1;
      }

      #pragma omp taskwait
    }
  }

  error=0;
  for (i=0; i<N; i++) {
    if (a[i] != i+1) printf("%d: error %d != %d, error %d\n", i, a[i], i+1, ++error);
  }
  if (! error) {
    printf("  test with nested maps and Parallel 1 thread conpleted successfully\n");
  } else {
    printf("  test with nested maps and Parallel 1 thread conpleted with %d error(s)\n", error);
    totError++;
  }
#endif

  printf("completed with %d errors\n", totError);
  return totError;
}

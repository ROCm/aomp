#include <stdio.h>
#define N (8*1024)
#define C 4
#define P 2
#define M (N/C)

#define TEST_PAR_NOWAIT      1
#define TEST_PAR_MAP_ALL     1

int a[N], b[N];

int main() {
  int i;
  int error, totError = 0;

#if TEST_PAR_NOWAIT
  for (i=0; i<N; i++) a[i] = b[i] = i;

  #pragma omp target data map(alloc: a, b)
  {  
    #pragma omp parallel for num_threads(P)
    for(int i=0; i<C; i++) {
      int lb = i*M;
      int ub = (i+1)*M;
      #pragma omp target nowait map(always to: b[lb:M]) map(always from: a[lb:M])
      {
        for(int j=lb; j<ub; j++) a[j] = b[j]+1;
      }
    } // par for
  } // data map

  error=0;
  for (i=0; i<N; i++) {
    if (a[i] != i+1) printf("%d: error %d != %d, error %d\n", i, a[i], i+1, ++error);
  }
  if (! error) {
    printf("  test with TEST_PAR_NOWAIT conpleted successfully\n");
  } else {
    printf("  test with TEST_PAR_NOWAIT conpleted with %d error(s)\n", error);
    totError++;
  }
#endif

#if TEST_PAR_MAP_ALL
  // map all
  for (i=0; i<N; i++) a[i] = b[i] = i;
  
  #pragma omp parallel for num_threads(P)
  for(int i=0; i<C; i++) {
    int lb = i*M;
    int ub = (i+1)*M;
    #pragma omp target nowait map(to: b[0:N]) map(always from: a[lb:M])
    {
      for(int j=lb; j<ub; j++) a[j] = b[j]+1;
    }
  } // par for
  

  error=0;
  for (i=0; i<N; i++) {
    if (a[i] != i+1) printf("%d: error %d != %d, error %d\n", i, a[i], i+1, ++error);
  }
  if (! error) {
    printf("  test with TEST_PAR_MAP_ALL conpleted successfully\n");
  } else {
    printf("  test with TEST_PAR_MAP_ALL conpleted with %d error(s)\n", error);
    totError++;
  }
#endif


  printf("completed with %d errors\n", totError);
  return totError;
}

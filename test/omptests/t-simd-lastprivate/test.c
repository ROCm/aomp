#include <stdio.h>

#include "../utilities/check.h"

#define N 100

int main()
{
  check_offloading();

  int a[N], aa[N];
  int i, error = 0;

  // initialize
  for(i=0; i<N; i++)
    aa[i] = a[i] = -1;

  // offload
  #pragma omp target map(tofrom: a[0:100]) 
  {  
    int k, n;
    #pragma omp simd lastprivate(n)
    for(k=0; k<N; k++) {
      a[k] = k;
      n = k;
    }
    a[0] = n;
  }

  // host
  for(i=0; i<N; i++)
    aa[i] = i;
  aa[0] = N-1;

  // check
  for(i=0; i<N; i++) {
    if (a[i] != aa[i]) 
      printf("%d: a %d != %d (error %d)\n", i, a[i], aa[i], ++error);
    if (error > 10) {
      printf("abort\n");
      return 0;
    }
  }

  // report
  printf("done with %d errors\n", error);
  return error;
}

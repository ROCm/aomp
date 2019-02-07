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
    int k;
    #pragma omp simd safelen(2)
    for(k=0; k<100; k++) {
      if (k > 1)
        a[k] = a[k-2] + 2;
      else
        a[k] = k;
    }
  }

  // host
  for(i=0; i<N; i++)
    aa[i] = i;

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

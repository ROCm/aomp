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
    int k, l;
    #pragma omp simd collapse(2)
    for(k=0; k<N/4; k++)
      for(l=0; l<4; l++)
        a[k*4+l] = k*4+l;
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

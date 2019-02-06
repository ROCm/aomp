#include <stdio.h>

#define N 100

int main()
{
  int a[N], aa[N];
  int i, error = 0;

  // initialize
  for(i=0; i<N; i++)
    aa[i] = a[i] = -1;

  // offload
  #pragma omp target map(tofrom: a[0:100]) 
  {
    int k;
    #pragma omp simd
    for(k=0; k<N; k++)
      a[k] = k;
  }

  // host
  for(i=0; i<N; i++)
    aa[i] = i;

  // check
  int first = -1;
  int last = - 1;
  for(i=0; i<N; i++) {
    if (a[i] != aa[i]) {
      if (first == -1) first = i;
      last = i;
      ++error;
    }
  }

  if (error) {
    if (error == 1)
      printf("one mismatch: [index:%d]: a %d != %d\n", first, a[first], aa[first]);
    else {
      printf("first mismatch: [index:%d]: a %d != %d (total errors: %d)\n", first, a[first], aa[first], error);
      printf("last mismatch: [index:%d]: a %d != %d (total errors %d)\n", last, a[last], aa[last], error);
   }
   return 0;
  }

  // report
  printf("Done with %d errors\n", error);
  return error;
}

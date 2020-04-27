#include <stdio.h>

#define N 3
#define M 4

int main()
{
  int a[N][M];
  int i,ii, error = 0; 
  // initialize
  for(i=0; i<N; i++)
    for(ii=0; ii<M; ii++)
      a[i][ii] = -1;

  // offload
  #pragma omp target map(tofrom: a[0:3][0:4]) 
  {
    int k,j;
    #pragma omp simd
    for(k=0; k<N; k++) {
      a[k][0] = k;
      #pragma omp simd
      for(j=0; j<M; j++) {
        a[k][j] = j;
      }
    }
  }

  // check
  for(i=0; i<N; i++) {
    for(ii=0; ii<M; ii++) {
      if (a[i][ii] != ii) {
        ++error;
      }
    }
  }

  // report
  printf("Done with %d errors\n", error);
  return error;
}

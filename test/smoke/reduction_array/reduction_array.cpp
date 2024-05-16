#include <stdio.h>
#define N 10

int main(){

  int i,j;
  float a[N], b[N];

  for(i=0; i<N; i++) a[i]=0.0e0;
  for(i=0; i<N; i++) b[i]=1.0e0+i;

#pragma omp target teams distribute parallel for reduction(+:a[0:N]) private(j)
  for(i=0; i<N; i++){
    for(j=0; j<N; j++){
      a[j] +=  b[j];
    }
  }

  printf(" a[0] a[N-1]: %f %f\n", a[0], a[N-1]);
  return 0;
}

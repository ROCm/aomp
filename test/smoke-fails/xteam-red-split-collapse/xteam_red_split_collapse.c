#include <stdio.h>
#include <omp.h>

int main()
{
  int N = 100;

  double a[N];
  double c[N][N];

  for (int i=0; i<N; i++)
    a[i]=i;

  double sum1 = 0;
  double tmp;
#pragma omp target teams map(tofrom:sum1) 
#pragma omp distribute parallel for reduction(+:sum1) collapse(2) private(tmp)
  for (int j = 0; j< N; j=j+1) {
    for (int i = 0; i < N; ++i) {
      sum1 += a[i];
      tmp = 0;
      for (int k = 0; k < N; ++k) {
	tmp += a[k];
      }
      c[i][j] = tmp;
    }
  }

  printf("sum1 = %f\n", sum1);
  printf("%f\n", c[10][20]);

  int rc = sum1 != 495000 || c[10][20] != 4950;
  
  if (!rc)
    printf("Success\n");

  return rc;
}

/// CHECK: DEVID:[[S:[ ]*]][[DEVID:[0-9]+]] SGN:8



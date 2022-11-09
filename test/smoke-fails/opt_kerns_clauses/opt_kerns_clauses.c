#include <stdio.h>
#include <omp.h>

int main()
{
  int N = 100000;

  double a[N], b[N];

  for (int i=0; i<N; i++) 
    a[i]=i;

  double sum1 = 0;

  int scalar1 = 10;
  
#pragma omp target teams distribute parallel for \
  defaultmap(firstprivate:scalar) defaultmap(to:aggregate) \
  map(from:b)
  for (int k = 0; k< N; k++)
    b[k]=a[k] * scalar1;
  
#pragma omp target teams distribute parallel for \
  defaultmap(firstprivate:scalar) defaultmap(to:aggregate) \
  map(tofrom:sum1) reduction(+:sum1)
  for (int j = 0; j< N; j=j+1)
    sum1 += a[j] + scalar1;

  printf("%f\n", sum1);

  int rc = sum1 != 5000950000;
  for (int i=0; i<N; i++)
    if (b[i] != a[i] * scalar1 ) {
      rc++;
      printf ("Wrong value: b[%d]=%f\n", i, b[i]);
    }
  if (!rc)
    printf("Success\n");

  return rc;
}

/// CHECK: DEVID:[[S:[ ]*]][[DEVID:[0-9]+]] SGN:4
/// CHECK: DEVID:[[S:[ ]*]][[DEVID:[0-9]+]] SGN:8


#include <stdio.h>
#include <omp.h>

int i;
static int j;

int main()
{
  int N = 10;
  double sum1 = 0;

  int a[N];
  int b[N];

  for (i=0; i<N; i++)
    b[i]=i;

  for (i=0; i<N; i++)
    a[i]=0;

#pragma omp target teams distribute parallel for map(a) map(b)
  for (i = 0; i< N; i++)
    a[i] = b[i];

#pragma omp target teams distribute parallel for
  for (j = 0; j< N; j++)
    a[j] = b[j];

#pragma omp target teams distribute parallel for map(tofrom:sum1) reduction(+:sum1) collapse(2)
  for (j = 0; j< N; j=j+1)
    for (i = 0; i < N; ++i)
      sum1 += b[i];

  printf("sum1 = %f\n", sum1);

  int rc = sum1 != 450;
  if (rc)
    return rc;
  
  for (i=0; i<N; i++)
    if (a[i] != b[i] ) {
      rc++;
      printf ("Wrong value: a[%d]=%d\n", i, a[i]);
    }

  if (!rc)
    printf("Success\n");

  return rc;
}

/// CHECK: DEVID:[[S:[ ]*]][[DEVID:[0-9]+]] SGN:4
/// CHECK: DEVID:[[S:[ ]*]][[DEVID:[0-9]+]] SGN:4
/// CHECK: DEVID:[[S:[ ]*]][[DEVID:[0-9]+]] SGN:8

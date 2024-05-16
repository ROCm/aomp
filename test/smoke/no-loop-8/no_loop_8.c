#include <stdio.h>
#include <omp.h>

int main()
{
  int N = 100000;

  int a[N];
  int b[N];

  int i;

  for (i=0; i<N; i++)
    b[i]=i;

  for (i=0; i<N; i++)
    a[i]=0;

  int j=0;
#pragma omp target teams distribute parallel for
  {
    for (j = 0; j< N; j+=3)
      a[j]=b[j];
  }

#pragma omp target teams distribute parallel for
  {
    for (j = 1; j< N; j=j+3)
      a[j]=b[j];
  }

#pragma omp target teams distribute parallel for
  {
    for (j = 2; j< N; j=3+j)
      a[j]=b[j];
  }

  int rc = 0;
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
/// CHECK: DEVID:[[S:[ ]*]][[DEVID:[0-9]+]] SGN:4

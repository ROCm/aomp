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

  int j;
#pragma omp target teams distribute parallel for schedule(auto)
  {
    for (j = 0; j< N; j++)
      a[j]=b[j];
  }
  
#pragma omp target teams distribute parallel for schedule(static, 1)
  {
    for (j = 0; j< N; j++)
      a[j]=b[j];
  }

#pragma omp target teams distribute parallel for schedule(static, 10)
  {
    for (j = 0; j< N; j++)
      a[j]=b[j];
  }
  
#pragma omp target teams distribute parallel for schedule(dynamic, 1)
  {
    for (j = 0; j< N; j++)
      a[j]=b[j];
  }
  
#pragma omp target teams distribute parallel for schedule(dynamic, 10)
  {
    for (j = 0; j< N; j++)
      a[j]=b[j];
  }
  
#pragma omp target teams distribute parallel for schedule(guided)
  {
    for (j = 0; j< N; j++)
      a[j]=b[j];
  }
  
#pragma omp target teams distribute parallel for schedule(runtime)
  {
    for (j = 0; j< N; j++)
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
/// CHECK: DEVID:[[S:[ ]*]][[DEVID:[0-9]+]] SGN:2
/// CHECK: DEVID:[[S:[ ]*]][[DEVID:[0-9]+]] SGN:4
/// CHECK: DEVID:[[S:[ ]*]][[DEVID:[0-9]+]] SGN:2
/// CHECK: DEVID:[[S:[ ]*]][[DEVID:[0-9]+]] SGN:2
/// CHECK: DEVID:[[S:[ ]*]][[DEVID:[0-9]+]] SGN:2

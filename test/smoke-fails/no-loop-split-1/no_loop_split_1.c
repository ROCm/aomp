#include <stdio.h>
#include <omp.h>

#pragma omp declare target
int foo(int i) { return i+1; }
#pragma omp end declare target

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

#pragma omp target 
#pragma omp teams distribute parallel for
    for (int k = 0; k< N; k++)
      a[k]=b[k];

#pragma omp target 
  {
#pragma omp teams distribute parallel for
    for (int k = 0; k< N; k++)
      a[k]=b[k];
  }

#pragma omp target 
#pragma omp teams distribute parallel for
    for (int k = 0; k< N/2; k+=2)
      a[k]=b[k];

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


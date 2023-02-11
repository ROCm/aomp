#include <stdio.h>
#include <omp.h>

int main()
{
  int N = 10;

  int a[N];
  int b[N];
  int c[N];

  int i;

  for (i=0; i<N; i++)
    b[i]=i;

  for (i=0; i<N; i++) {
    a[i] = 0;
    c[i] = 0;
  }

#pragma omp target teams
#pragma omp distribute parallel for
    {
      for (int k = 0; k< N; k++) {
	a[k]=b[k];
      }
  }

#pragma omp target 
#pragma omp teams
#pragma omp distribute parallel for
  {
    {
      for (int k = 0; k< N; k++) {
	c[k]=b[k];
      }
    }
  }

#pragma omp target 
#pragma omp teams
#pragma omp distribute parallel for
  for (int k = 0; k< N; k++) {
    c[k]=b[k];
  }

#pragma omp target 
#pragma omp teams
  {
#pragma omp distribute parallel for    
    for (int k = 0; k< N; k++)
      a[k]=b[k];
#pragma omp distribute parallel for    
    for (int k = 0; k< N; k+=2)
      c[k]=b[k];
  }
  
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
  for (i=0; i<N; i++) {
    if (a[i] != b[i] || c[i] != b[i]) {
      rc++;
      printf ("Wrong value: a[%d]=%d c[%d]=%d\n", i, a[i], i, c[i]);
    }
  }
  if (!rc)
    printf("Success\n");

  return rc;
}

/// CHECK: DEVID:[[S:[ ]*]][[DEVID:[0-9]+]] SGN:5
/// CHECK: DEVID:[[S:[ ]*]][[DEVID:[0-9]+]] SGN:5
/// CHECK: DEVID:[[S:[ ]*]][[DEVID:[0-9]+]] SGN:5
/// CHECK: DEVID:[[S:[ ]*]][[DEVID:[0-9]+]] SGN:3
/// CHECK: DEVID:[[S:[ ]*]][[DEVID:[0-9]+]] SGN:5
/// CHECK: DEVID:[[S:[ ]*]][[DEVID:[0-9]+]] SGN:5
/// CHECK: DEVID:[[S:[ ]*]][[DEVID:[0-9]+]] SGN:5


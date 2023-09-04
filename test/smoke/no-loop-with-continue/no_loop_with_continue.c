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

#pragma omp target teams distribute parallel for
  for (j = 0; j< N; j++) {
    if (j < 10) continue;
    a[j]=b[j];
  }
  
#pragma omp target teams distribute parallel for
  for (j = 0; j< N; j++) {
    for (i = 0; i < N; ++i) {
      if (i < 10) continue;
      a[i]=b[i];
    }
  }

  int rc = 0;
  for (i=0; i<N; i++)
    if (i < 10 && a[i] != 0) {
      rc++;
      printf("1:Wrong value: a[%d]=%d\n", i, a[i]);
    } else if (i >=10 && a[i] != b[i] ) {
      rc++;
      printf("2:Wrong value: a[%d]=%d\n", i, a[i]);
    }

  if (!rc)
    printf("Success\n");

  return rc;
}

/// CHECK: DEVID:[[S:[ ]*]][[DEVID:[0-9]+]] SGN:4


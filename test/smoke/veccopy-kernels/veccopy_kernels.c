#include <stdio.h>
#include <omp.h>

int main()
{
  int N = 100000;

  int a[N];
  int b[N];
  int c[N];

  int i;

  for (i=0; i<N; i++)
    a[i]=0;

  for (i=0; i<N; i++)
    b[i]=i;

  for (i=0; i<N; i++)
    c[i]=i;

#pragma omp target parallel for
  {
    for (int j = 0; j< N; j++)
      a[j]=b[j];
  }

#pragma omp target teams distribute parallel for
  {
    for (int j = 0; j< N; j++)
      c[j]=b[j];
  }

  int rc = 0;
  for (i=0; i<N; i++)
    if (a[i] != b[i] || c[i] != b[i]) {
      rc++;
      printf ("Wrong value: a[%d]=%d c[%d]=%d\n", i, a[i], i, c[i]);
    }

  if (!rc)
    printf("Success\n");

  return rc;
}


#include <stdio.h>
#include <omp.h>

// Cause the compiler to set amdhsa_uses_dynamic_stack to '1' using recursion.
// That is: stack requirement for main's target region may not be calculated.

// This recursive function will eventually return 0.
int recursiveFunc(const int Recursions) {
  if (Recursions < 1)
    return 0;

  int j[Recursions];
#pragma omp target private(j)
  {
    ;
  }

  return recursiveFunc(Recursions - 1);
}

int main()
{
  int N = 256;

  int a[N];
  int b[N];

  int i;

  for (i=0; i<N; i++)
    a[i]=0;

  for (i=0; i<N; i++)
    b[i]=i;

#pragma omp target parallel for
  {
    for (int j = 0; j< N; j++)
      a[j] = b[j] + recursiveFunc(j);
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


/*
 * Test whether OpenMP directives work in a global constructor
 * used in a shared library. 
 * This currently fails with the following assertion.
 * Needs investigation.
omptarget.cpp:1617: int target(ident_t *, DeviceTy &, void *, KernelArgsTy &, AsyncInfoTy &): Assertion `TargetTable && "Global data has not been mapped\n"' failed.
 */

#include <stdio.h>
#include <omp.h>

extern int status;
int main()
{
  int N = 10;

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
      a[j]=b[j];
  }

  int rc = 0;
  for (i=0; i<N; i++)
    if (a[i] != b[i] ) {
      rc++;
      printf ("Wrong varlue: a[%d]=%d\n", i, a[i]);
    }

  if (!rc && !status)
    printf("Success\n");

  return rc;
}

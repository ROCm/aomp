/*
 * Test min/max reduction using fmin/fmax but multiple occurences 
 * of the reduction variables and in nested scopes.
 */
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <omp.h>

int main()
{
  int N = 1000;

  int a[N];
  int b[N][N];

  for (int i = 0; i < N; i++)
  {
    a[i] = i + 11;
    for (int j = 0; j < N; ++j)
      b[i][j] = i + j + 3;
  }

  int max1 = 0;
  int min1 = 1000000;

#pragma omp target teams distribute parallel for reduction(max : max1) reduction(min : min1)
  for (int i = 0; i < N; i = i + 1)
  {
    max1 = fmax(max1, a[i]);
    min1 = fmin(min1, a[i]);
    for (int j = 0; j < N; ++j)
    {
      max1 = fmax(b[i][j], max1);
      min1 = fmin(min1, b[i][j]);
    }
  }

  printf("max1 = %d min1 = %d\n", max1, min1);
  int rc = (max1 != 2001) || (min1 != 3);

  if (!rc)
    printf("Success\n");
  else
    printf("Failed\n");

  return rc;
}

// Cross-team reductions are emitted as plain SPMD kernels (SGN:2); the
// downstream Xteam reduction execution mode (SGN:8) has been removed.
/// CHECK: DEVID:[[S:[ ]*]][[DEVID:[0-9]+]] SGN:2

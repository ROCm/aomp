/*
 * Test min/max reduction on floats using minf/maxf.
 */
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <omp.h>

int main()
{
  int N = 1000;

  float a[N];

  for (int i = 0; i < N; i++)
    a[i] = i + 11;

  float max1 = 0;
  float min1 = 1000000;

#pragma omp target teams distribute parallel for reduction(max : max1) reduction(min : min1)
  for (int i = 0; i < N; i = i + 1)
  {
    max1 = fmaxf(max1, a[i]);
    min1 = fminf(min1, a[i]);
  }

  printf("max1 = %f min1 = %f\n", max1, min1);
  int rc = (max1 != 1010) || (min1 != 11);

  if (!rc)
    printf("Success\n");
  else
    printf("Failed\n");

  return rc;
}



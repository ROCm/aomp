/*
 * Test min/max reduction on long double. It fails since reduction
 * on long double is not supported on the GPU.
 */
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <omp.h>

int main()
{
  int N = 1000;

  long double a[N];

  for (int i = 0; i < N; i++)
    a[i] = i + 11;

  long double max1 = 0;
  long double min1 = 1000000;

#pragma omp target teams distribute parallel for reduction(max : max1) reduction(min : min1)
  for (int i = 0; i < N; i = i + 1)
  {
    max1 = fmaxl(max1, a[i]);
    min1 = fminl(min1, a[i]);
  }

  printf("max1 = %Lf min1 = %Lf\n", max1, min1);
  int rc = (max1 != 1010) || (min1 != 11);

  if (!rc)
    printf("Success\n");
  else
    printf("Failed\n");

  return rc;
}



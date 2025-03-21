/*
 * Test min/max/sum reduction when fast Xteam reduction is enabled. In the same kernel,
 * min/max and sum reductions are present. Xteam reduction will not be enabled in this kernel.
 * But in some other kernel, Xteam reduction can be used.
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
  float sum1 = 0;
  float sum2 = 0;

#pragma omp target teams distribute parallel for reduction(max : max1) reduction(min : min1) reduction(+ : sum1)
  for (int i = 0; i < N; i = i + 1)
  {
    max1 = fmaxf(max1, a[i]);
    min1 = fminf(min1, a[i]);
    sum1 += a[i];
  }

#pragma omp target teams distribute parallel for reduction(+ : sum2)
  for (int i = 0; i < N; i = i + 1)
    sum2 += a[i];

  printf("max1 = %f min1 = %f sum1 = %f sum2 = %f\n", max1, min1, sum1, sum2);
  int rc = (max1 != 1010) || (min1 != 11) || (sum1 != 510500) || (sum2 != 510500);

  if (!rc)
    printf("Success\n");
  else
    printf("Failed\n");

  return rc;
}



/*
 * Test min/max reduction using fmin/fmax with unrelated calls in
 * the kernels. Compile using -fopenmp-target-fast, that will enable
 * Xteam reduction even with calls inside.
*/
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <omp.h>

int main()
{
  int N = 10000;

  int a[N];

  for (int i=0; i<N; i++)
    a[i]=i+11;

  int max1, max2, max3;
  max1 = max2 = max3 = 0;
  int min1, min2, min3;
  min1 = min2 = min3 = 1000000;

#pragma omp target teams distribute parallel for reduction(max : max1)
  for (int j = 0; j < N; j = j + 1)
  {
    a[j] = abs(a[j]);
    max1 = fmax(max1, a[j]);
  }

#pragma omp target teams distribute parallel for reduction(max : max2)
  for (int j = 0; j < N; j = j + 1)
    max2 = abs((int)fmax(max2, a[j]));

#pragma omp target teams distribute parallel for reduction(max : max3)
  for (int j = 1; j < N; j = j + 1)
    max3 = fmax(max3, abs(a[j]));

#pragma omp target teams distribute parallel for reduction(min : min1)
  for (int j = 0; j < N; j = j + 1)
  {
    a[j] = abs(a[j]);
    min1 = fmin(min1, a[j]);
  }

#pragma omp target teams distribute parallel for reduction(min : min2)
  for (int j = 0; j < N; j = j + 1)
    min2 = abs((int)fmin(min2, a[j]));

#pragma omp target teams distribute parallel for reduction(min : min3)
  for (int j = 1; j < N; j = j + 1)
    min3 = fmin(min3, abs(a[j]));

    printf("max1 = %d max2 = %d max3 = %d\n", max1, max2, max3);
    printf("min1 = %d min2 = %d min3 = %d\n", min1, min2, min3);

  int rc = (max1 != 10010) || (max2 != 10010) || (max3 != 10010) || (min1 != 11) || (min2 != 11) || (min3 != 12);

  if (!rc)
    printf("Success\n");
  else
    printf("Failed\n");

  return rc;
}

/// CHECK: DEVID:[[S:[ ]*]][[DEVID:[0-9]+]] SGN:8
/// CHECK: DEVID:[[S:[ ]*]][[DEVID:[0-9]+]] SGN:2
/// CHECK: DEVID:[[S:[ ]*]][[DEVID:[0-9]+]] SGN:8
/// CHECK: DEVID:[[S:[ ]*]][[DEVID:[0-9]+]] SGN:8
/// CHECK: DEVID:[[S:[ ]*]][[DEVID:[0-9]+]] SGN:2
/// CHECK: DEVID:[[S:[ ]*]][[DEVID:[0-9]+]] SGN:8
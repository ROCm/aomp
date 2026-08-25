/*
 * Test min/max reduction using fmin/fmax with unrelated calls in
 * the kernels. Compile using -fopenmp-target-fast, that will enable
 * Xteam reduction even with calls inside.
*/
#include <stdio.h>
#include <math.h>
#include <omp.h>

int main()
{
  int N = 10000;

  double a[N];

  for (int i=0; i<N; i++)
    a[i]=i+11;

  double max1, max2, max3;
  max1 = max2 = max3 = 0;
  double min1, min2, min3;
  min1 = min2 = min3 = 1000000;

#pragma omp target teams distribute parallel for reduction(max : max1)
  for (int j = 0; j < N; j = j + 1)
  {
    a[j] = fabs(a[j]);
    max1 = fmax(max1, a[j]);
  }

#pragma omp target teams distribute parallel for reduction(max : max2)
  for (int j = 0; j < N; j = j + 1)
    max2 = fabs(fmax(max2, a[j]));

#pragma omp target teams distribute parallel for reduction(max : max3)
  for (int j = 1; j < N; j = j + 1)
    max3 = fmax(max3, fabs(a[j]));

#pragma omp target teams distribute parallel for reduction(min : min1)
  for (int j = 0; j < N; j = j + 1)
  {
    a[j] = fabs(a[j]);
    min1 = fmin(min1, a[j]);
  }

#pragma omp target teams distribute parallel for reduction(min : min2)
  for (int j = 0; j < N; j = j + 1)
    min2 = fabs(fmin(min2, a[j]));

#pragma omp target teams distribute parallel for reduction(min : min3)
  for (int j = 1; j < N; j = j + 1)
    min3 = fmin(min3, fabs(a[j]));

    printf("max1 = %f max2 = %f max3 = %f\n", max1, max2, max3);
    printf("min1 = %f min2 = %f min3 = %f\n", min1, min2, min3);

  int rc = (max1 != 10010) || (max2 != 10010) || (max3 != 10010) || (min1 != 11) || (min2 != 11) || (min3 != 12);

  if (!rc)
    printf("Success\n");
  else
    printf("Failed\n");

  return rc;
}

// Cross-team reductions are emitted as plain SPMD kernels (SGN:2); the
// downstream Xteam reduction execution mode (SGN:8) has been removed.
/// CHECK: DEVID:[[S:[ ]*]][[DEVID:[0-9]+]] SGN:2
/// CHECK: DEVID:[[S:[ ]*]][[DEVID:[0-9]+]] SGN:2
/// CHECK: DEVID:[[S:[ ]*]][[DEVID:[0-9]+]] SGN:2
/// CHECK: DEVID:[[S:[ ]*]][[DEVID:[0-9]+]] SGN:2
/// CHECK: DEVID:[[S:[ ]*]][[DEVID:[0-9]+]] SGN:2
/// CHECK: DEVID:[[S:[ ]*]][[DEVID:[0-9]+]] SGN:2


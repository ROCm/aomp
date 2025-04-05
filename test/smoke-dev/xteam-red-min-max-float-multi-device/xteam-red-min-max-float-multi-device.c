/*
 * Test multi-device min/max reduction on floats using minf/maxf.
 * There are 2 target regions in this program, the first has min/max reduction
 * and the second has a sum reduction. The program is compiled with multi-device
 * ON. Since multi-device compilation may be incompatible with Xteam min/max, the
 * first target region does not use Xteam reduction. The second one, however, does.
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

#pragma omp target teams distribute parallel for reduction(max : max1) reduction(min : min1)
  for (int i = 0; i < N; i = i + 1)
  {
    max1 = fmaxf(max1, a[i]);
    min1 = fminf(min1, a[i]);
  }

  #pragma omp target teams distribute parallel for reduction(+ : sum1)
  for (int i = 0; i < N; i = i + 1)
    sum1 += a[i];

  printf("max1 = %f min1 = %f sum1 = %f\n", max1, min1, sum1);
  int rc = (max1 != 1010) || (min1 != 11) || (sum1 != 510500);

  if (!rc)
    printf("Success\n");
  else
    printf("Failed\n");

  return rc;
}

/// CHECK: DEVID:[[S:[ ]*]][[DEVID:[0-9]+]] SGN:2
/// CHECK: DEVID:[[S:[ ]*]][[DEVID:[0-9]+]] SGN:8
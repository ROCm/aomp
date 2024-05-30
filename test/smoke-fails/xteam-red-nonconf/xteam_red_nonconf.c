#include <omp.h>
#include <stdio.h>

int main() {
  int N = 10;

  double a[N];

  for (int i = 0; i < N; i++)
    a[i] = i;

  double sum1, sum2, sum3, sum4;
  sum1 = sum2 = sum3 = sum4 = 0;

#pragma omp target teams distribute parallel for reduction(+ : sum1)
  for (int j = 0; j < N; j = j + 1)
    sum1 += sum1 + a[j];

#pragma omp target teams distribute parallel for reduction(+ : sum2)
  for (int j = 0; j < N; j = j + 2)
    sum2 = sum2 + a[j] + sum2;

#pragma omp target teams distribute parallel for reduction(+ : sum3)
  for (int j = 0; j < N; j = j + 1) {
    sum3 = 2 * sum3 + a[j];
  }

#pragma omp target teams distribute parallel for reduction(+ : sum4)
  for (int j = 0; j < N; j = j + 1) {
    sum4 = a[j];
  }

  printf("%f %f %f %f\n", sum1, sum2, sum3, sum4);

  int rc = (sum1 != 1013) || (sum2 != 52) || (sum3 != 1013) || (sum4 != 9);

  if (!rc)
    printf("Success\n");

  return rc;
}

/// CHECK: DEVID:[[S:[ ]*]][[DEVID:[0-9]+]] SGN:2
/// CHECK: DEVID:[[S:[ ]*]][[DEVID:[0-9]+]] SGN:2
/// CHECK: DEVID:[[S:[ ]*]][[DEVID:[0-9]+]] SGN:2
/// CHECK: DEVID:[[S:[ ]*]][[DEVID:[0-9]+]] SGN:2

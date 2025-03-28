/* Test passing reduction variable as reference to a function. 
 * Try a few combinations of operators. 
 */
#include <stdio.h>
#include <math.h>
#include <omp.h>

void compute_min(int j, double &my_min, double a[]) {
  my_min = fmin(my_min, a[j]);
}

void compute_max(int j, double &my_max, double a[]) {
  my_max = fmax(my_max, a[j]);
}

void compute_sum(int j, double &my_sum, double a[]) {
  my_sum += a[j];
}

int main()
{
  int N = 10000;

  double a[N];

  for (int i=0; i<N; i++)
    a[i]=i+3;

  double sum1, min1, max1;
  sum1 = max1 = 0;
  min1 = 100000;

  int rc = 0;

#pragma omp target teams distribute parallel for reduction(min : min1)
  for (int j = 0; j < N; j = j + 1)
    compute_min(j, min1, a);
printf("min1=%f\n", min1);
rc = min1 != 3;

#pragma omp target teams distribute parallel for reduction(max : max1)
  for (int j = 0; j < N; j = j + 1)
    compute_max(j, max1, a);
printf("max1=%f\n", max1);
rc = max1 != 10002;

max1 = 0;
min1 = 100000;
#pragma omp target teams distribute parallel for reduction(min : min1) reduction(max : max1)
for (int j = 0; j < N; j = j + 1)
{
  compute_min(j, min1, a);
  compute_max(j, max1, a);
}
printf("min1=%f max1=%f\n", min1, max1);
rc = (min1 != 3) || (max1 != 10002);

min1 = 100000;
#pragma omp target teams distribute parallel for reduction(+ : sum1) reduction(min : min1)
for (int j = 0; j < N; j = j + 1)
{
  compute_sum(j, sum1, a);
  compute_min(j, min1, a);
}
printf("sum1=%f min1=%f\n", sum1, min1);
rc = (sum1 != 50025000) || (min1 != 3);

sum1 = max1 = 0;
#pragma omp target teams distribute parallel for reduction(+ : sum1) reduction(max : max1)
for (int j = 0; j < N; j = j + 1)
{
  compute_sum(j, sum1, a);
  compute_max(j, max1, a);
}
printf("sum1=%f max1=%f\n", sum1, max1);
rc = (sum1 != 50025000) || (max1 != 10002);

if (!rc)
  printf("Success\n");
else
  printf("Failed\n");

return rc;
}



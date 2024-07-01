#include <stdio.h>
#include <omp.h>

int compute_sum_res(int j, double &result, double a[]) {
  result += a[j];
  return 1;
}

void compute_sum(int j, double &result, double a[]) {
  result += a[j];
}

double compute_sum_rval(int j, double rval, double a[]) {
  return rval + a[j];
}

int foo(int i) { return 2*i; }

int main()
{
  int N = 10000;

  double a[N];

  for (int i=0; i<N; i++)
    a[i]=i;

  double sum1, sum2, sum3, sum4, sum5;
  sum1 = sum2 = sum3 = sum4 = sum5 = 0;

  int res = 0;
#pragma omp target teams distribute parallel for reduction(+:sum1) map(tofrom:res)
  for (int j = 0; j< N; j=j+1)
    res = compute_sum_res(j, sum1, a);

#pragma omp target teams distribute parallel for reduction(+:sum2)
  for (int j = 0; j< N; j=j+1)
    compute_sum(j, sum2, a);

#pragma omp target teams distribute parallel for reduction(+:sum3)
  for (int j = 0; j< N; j=j+1)
    sum3 = compute_sum_rval(j, sum3, a);

#pragma omp target teams distribute parallel for reduction(+:sum4)
  for (int j = 0; j< N; j=j+1)
    foo(compute_sum_res(j, sum4, a));

#pragma omp target teams distribute parallel for reduction(+:sum5)
  for (int j = 0; j< N; j=j+1)
    compute_sum_res(j, sum5, a);

  printf("%f %f %f %f %f\n", sum1, sum2, sum3, sum4, sum5);
  
  int rc =
    (sum1 != 49995000) || (sum2 != 49995000) || (sum3 != 49995000) ||
    (sum4 != 49995000) || (sum5 != 49995000);

  if (!rc)
    printf("Success\n");

  return rc;
}

/// CHECK: DEVID:[[S:[ ]*]][[DEVID:[0-9]+]] SGN:2
/// CHECK: DEVID:[[S:[ ]*]][[DEVID:[0-9]+]] SGN:2
/// CHECK: DEVID:[[S:[ ]*]][[DEVID:[0-9]+]] SGN:2
/// CHECK: DEVID:[[S:[ ]*]][[DEVID:[0-9]+]] SGN:2
/// CHECK: DEVID:[[S:[ ]*]][[DEVID:[0-9]+]] SGN:2


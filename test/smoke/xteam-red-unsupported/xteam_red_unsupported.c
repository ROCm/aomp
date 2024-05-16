#include <stdio.h>
#include <math.h>
#include <omp.h>

int main()
{
  int N = 1000000;

  double a[N];

  for (int i=0; i<N; i++)
    a[i]=i;

  double sum1, my_max, my_min;
  sum1 = my_max = 0;
  my_min = N;

  // This is supported by Xteam reduction 
#pragma omp target teams distribute parallel for map(tofrom:sum1) reduction(+:sum1)
  for (int j = 0; j< N; j=j+1)
    sum1 += 3*a[j];

  // The rest are not supported
#pragma omp target teams distribute parallel for map(tofrom:sum1) reduction(-:sum1)
  for (int j = 0; j< N; j=j+1)
    sum1 -= a[j];

#pragma omp target teams distribute parallel for map(tofrom:sum1) reduction(-:sum1)
  for (int j = 0; j< N; j=j+1)
    sum1 = sum1 - a[j];

#pragma omp target teams distribute parallel for map(tofrom:my_max) reduction(max:my_max)
  for (int j = 0; j< N; j=j+1)
    my_max = fmax(my_max, a[j]);

#pragma omp target teams distribute parallel for map(tofrom:my_min) reduction(min:my_min)
  for (int j = 1; j< N; j=j+1)
    my_min = fmin(my_min, a[j]);
  
  printf("sum1 = %f my_max = %f my_min = %f\n", sum1, my_max, my_min);
  
  int rc = sum1 != 499999500000 || my_max != N-1 || my_min != 1;

  if (!rc)
    printf("Success\n");

  return rc;
}

/// CHECK: DEVID:[[S:[ ]*]][[DEVID:[0-9]+]] SGN:8
/// CHECK: DEVID:[[S:[ ]*]][[DEVID:[0-9]+]] SGN:2
/// CHECK: DEVID:[[S:[ ]*]][[DEVID:[0-9]+]] SGN:2
/// CHECK: DEVID:[[S:[ ]*]][[DEVID:[0-9]+]] SGN:2
/// CHECK: DEVID:[[S:[ ]*]][[DEVID:[0-9]+]] SGN:2


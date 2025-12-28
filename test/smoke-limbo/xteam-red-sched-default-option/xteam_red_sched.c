#include <stdio.h>
#include <omp.h>

int main()
{
  int N = 100;

  double a[N];

  for (int i=0; i<N; i++)
    a[i]=i;

  double sum1 = 0;
  
#pragma omp target teams distribute parallel for map(tofrom:sum1) reduction(+:sum1) schedule(auto)
  for (int j = 0; j< N; j=j+1)
    sum1 += a[j];

#pragma omp target teams distribute parallel for map(tofrom:sum1) reduction(+:sum1) schedule(static,1)
  for (int j = 0; j< N; j=j+1)
    sum1 += a[j];

#pragma omp target teams distribute parallel for map(tofrom:sum1) reduction(+:sum1) schedule(static,10)
  for (int j = 0; j< N; j=j+1)
    sum1 += a[j];

#pragma omp target teams distribute parallel for map(tofrom:sum1) reduction(+:sum1) schedule(dynamic,1)
  for (int j = 0; j< N; j=j+1)
    sum1 += a[j];

#pragma omp target teams distribute parallel for map(tofrom:sum1) reduction(+:sum1) schedule(dynamic,10)
  for (int j = 0; j< N; j=j+1)
    sum1 += a[j];

#pragma omp target teams distribute parallel for map(tofrom:sum1) reduction(+:sum1) schedule(guided)
  for (int j = 0; j< N; j=j+1)
    sum1 += a[j];

#pragma omp target teams distribute parallel for map(tofrom:sum1) reduction(+:sum1) schedule(runtime)
  for (int j = 0; j< N; j=j+1)
    sum1 += a[j];

  printf("%f\n", sum1);
  
  int rc = sum1 != 34650;

  if (!rc)
    printf("Success\n");

  return rc;
}

/// CHECK: DEVID:[[S:[ ]*]][[DEVID:[0-9]+]] SGN:8
/// CHECK: DEVID:[[S:[ ]*]][[DEVID:[0-9]+]] SGN:8
/// CHECK: DEVID:[[S:[ ]*]][[DEVID:[0-9]+]] SGN:2
/// CHECK: DEVID:[[S:[ ]*]][[DEVID:[0-9]+]] SGN:8
/// CHECK: DEVID:[[S:[ ]*]][[DEVID:[0-9]+]] SGN:2
/// CHECK: DEVID:[[S:[ ]*]][[DEVID:[0-9]+]] SGN:2
/// CHECK: DEVID:[[S:[ ]*]][[DEVID:[0-9]+]] SGN:2


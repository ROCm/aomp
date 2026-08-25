#include <stdio.h>
#include <omp.h>

int main()
{
  int N = 100;

  double a[N], b[N];

  for (int i=0; i<N; i++)
    a[i]=i;

  double sum1 = 0;

#pragma omp target teams distribute parallel for map(tofrom:sum1) reduction(+:sum1)
  for (int j = 0; j< N; j=j+1)
    sum1 = a[j] + sum1;
  
#pragma omp target teams distribute parallel for map(tofrom:sum1) reduction(+:sum1)
  for (int j = 0; j< N; j=j+1)
    sum1 = sum1 + a[j];

#pragma omp target teams distribute parallel for map(tofrom:sum1) reduction(+:sum1)
  for (int j = 0; j< N; j=j+1)
    sum1 = a[j] + sum1 + a[j];
  
  printf("sum1 = %f\n", sum1);
  
  int rc = sum1 != 19800;
  
  if (!rc)
    printf("Success\n");

  return rc;
}

// Cross-team reductions are emitted as plain SPMD kernels (SGN:2); the
// downstream Xteam reduction execution mode (SGN:8) has been removed.
/// CHECK: DEVID:[[S:[ ]*]][[DEVID:[0-9]+]] SGN:2
/// CHECK: DEVID:[[S:[ ]*]][[DEVID:[0-9]+]] SGN:2
/// CHECK: DEVID:[[S:[ ]*]][[DEVID:[0-9]+]] SGN:2



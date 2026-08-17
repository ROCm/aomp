#include <stdio.h>
#include <omp.h>

int main()
{
  int N = 1000000;

  double a[N];

  for (int i=0; i<N; i++)
    a[i]=i;

  double sum1, sum2, sum3, sum4;
  sum1 = sum2 = sum3 = sum4 = 0;
  
#pragma omp target teams distribute parallel for reduction(+:sum1) num_teams(1)
  for (int j = 0; j< N; j=j+1)
    sum1 += a[j];

#pragma omp target teams distribute parallel for reduction(+:sum2) num_teams(50)
  for (int j = 0; j< N; j=j+1)
    sum2 += a[j];

#pragma omp target teams distribute parallel for reduction(+:sum3) num_teams(40)
  for (int j = 0; j< N; j=j+1)
    sum3 += a[j];

#pragma omp target teams distribute parallel for reduction(+:sum4) num_teams(10)
  for (int j = 0; j< N; j=j+1)
    sum4 += a[j];

  printf("%f %f %f %f\n", sum1, sum2, sum3, sum4);
  
  int rc =
    (sum1 != 499999500000) ||
    (sum2 != 499999500000) ||
    (sum3 != 499999500000) ||
    (sum4 != 499999500000);

  if (!rc)
    printf("Success\n");
  
  return rc;
}

// Cross-team reductions are emitted as plain SPMD kernels (SGN:2); the
// downstream Xteam reduction execution mode (SGN:8) has been removed. The
// kernels no longer carry the two extra Xteam arguments either, hence
// 'args: 5' instead of the former 'args: 7'. The num_teams clause is still
// honored verbatim.
/// CHECK: DEVID:[[S:[ ]*]][[DEVID:[0-9]+]] SGN:2 ConstWGSize:512 args: 5 teamsXthrds:( 1X 512)
/// CHECK: DEVID:[[S:[ ]*]][[DEVID:[0-9]+]] SGN:2 ConstWGSize:512 args: 5 teamsXthrds:( 50X 512)
/// CHECK: DEVID:[[S:[ ]*]][[DEVID:[0-9]+]] SGN:2 ConstWGSize:512 args: 5 teamsXthrds:( 40X 512)
/// CHECK: DEVID:[[S:[ ]*]][[DEVID:[0-9]+]] SGN:2 ConstWGSize:512 args: 5 teamsXthrds:( 10X 512)

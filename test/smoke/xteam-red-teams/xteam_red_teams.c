#include <stdio.h>
#include <omp.h>

int main()
{
  int N = 1000000;

  double a[N];

  for (int i=0; i<N; i++)
    a[i]=i;

  double sum1, sum2;
  sum1 = sum2 = 0;
  
#pragma omp target teams distribute parallel for reduction(+:sum1) num_teams(1)
  for (int j = 0; j< N; j=j+1)
    sum1 += a[j];

#pragma omp target teams distribute parallel for reduction(+:sum2) num_teams(5)
  for (int j = 0; j< N; j=j+1)
    sum2 += a[j];

  printf("%f %f\n", sum1, sum2);
 
  int rc =
    (sum1 != 499999500000) ||
    (sum2 != 499999500000);

  if (!rc)
    printf("Success\n");
  
  return rc;
}

/// CHECK: DEVID:[[S:[ ]*]][[DEVID:[0-9]+]] SGN:8 ConstWGSize:1024 args: 7 teamsXthrds:(   1X1024) 
/// CHECK: DEVID:[[S:[ ]*]][[DEVID:[0-9]+]] SGN:8 ConstWGSize:1024 args: 7 teamsXthrds:(  5X1024)


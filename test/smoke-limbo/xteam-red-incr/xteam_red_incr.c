#include <stdio.h>
#include <omp.h>

int main()
{
  int N = 10;
  int sum = 0;
  
#pragma omp target teams distribute parallel for reduction(+:sum)
  for (int j = 0; j< N; j=j+1)
    sum++;

  printf("sum = %d\n", sum);
  int rc = sum != 10;

  if (!rc)
    printf("Success\n");

  return rc;
}

/// CHECK: DEVID:[[S:[ ]*]][[DEVID:[0-9]+]] SGN:2

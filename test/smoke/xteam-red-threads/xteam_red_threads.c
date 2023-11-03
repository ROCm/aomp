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
  
#pragma omp target teams distribute parallel for map(tofrom:sum1) reduction(+:sum1) thread_limit(128)
  for (int j = 0; j< N; j=j+1)
    sum1 += a[j];

#pragma omp target teams distribute parallel for map(tofrom:sum2) reduction(+:sum2) thread_limit(512)
  for (int j = 0; j< N; j=j+1)
    sum2 += a[j];

#pragma omp target teams distribute parallel for map(tofrom:sum3) reduction(+:sum3) thread_limit(128) num_threads(256)
  for (int j = 0; j< N; j=j+1)
    sum3 += a[j];

#pragma omp target teams distribute parallel for map(tofrom:sum4) reduction(+:sum4) thread_limit(512) num_threads(128)
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
  
  sum1 = sum2 = sum3 = sum4 = 0;

#pragma omp target teams distribute parallel for map(tofrom:sum1) reduction(+:sum1) thread_limit(128)
  for (int j = 0; j< N; j=j+2)
    sum1 += a[j];

#pragma omp target teams distribute parallel for map(tofrom:sum2) reduction(+:sum2) thread_limit(512)
  for (int j = 0; j< N; j=j+2)
    sum2 += a[j];

#pragma omp target teams distribute parallel for map(tofrom:sum3) reduction(+:sum3) thread_limit(128) num_threads(256)
  for (int j = 0; j< N; j=j+2)
    sum3 += a[j];

#pragma omp target teams distribute parallel for map(tofrom:sum4) reduction(+:sum4) thread_limit(512) num_threads(128)
  for (int j = 0; j< N; j=j+2)
    sum4 += a[j];

  printf("%f %f %f %f\n", sum1, sum2, sum3, sum4);
  
  rc =
    (sum1 != 249999500000) ||
    (sum2 != 249999500000) ||
    (sum3 != 249999500000) ||
    (sum4 != 249999500000);

  if (!rc)
    printf("Success\n");
  
  sum1 = sum2 = sum3 = sum4 = 0;

#pragma omp target teams distribute parallel for map(tofrom:sum1,sum2) reduction(+:sum1,sum2) thread_limit(128)
  for (int j = 0; j< N; j=j+1) {
    sum1 += a[j];
    sum2 += a[j];
  }

#pragma omp target teams distribute parallel for map(tofrom:sum3,sum4) reduction(+:sum3,sum4) thread_limit(512)
  for (int j = 0; j< N; j=j+1) {
    sum3 += a[j];
    sum4 += a[j];
  }

#pragma omp target teams distribute parallel for map(tofrom:sum1,sum2) reduction(+:sum1,sum2) thread_limit(128) num_threads(256)
  for (int j = 0; j< N; j=j+1) {
    sum1 += a[j];
    sum2 += a[j];
  }

#pragma omp target teams distribute parallel for map(tofrom:sum3,sum4) reduction(+:sum3,sum4) thread_limit(512) num_threads(128)
  for (int j = 0; j< N; j=j+1) {
    sum3 += a[j];
    sum4 += a[j];
  }
  
  printf("%f %f %f %f\n", sum1, sum2, sum3, sum4);
  
  rc =
    (sum1 != 999999000000) ||
    (sum2 != 999999000000) ||
    (sum3 != 999999000000) ||
    (sum4 != 999999000000);

  if (!rc)
    printf("Success\n");

  return rc;
}

/// CHECK: DEVID:[[S:[ ]*]][[DEVID:[0-9]+]] SGN:8 ConstWGSize:128  args: 7 teamsXthrds:([[S:[ ]*]][[NUM_TEAMS:[0-9]+]]X 128)
/// CHECK: DEVID:[[S:[ ]*]][[DEVID:[0-9]+]] SGN:8 ConstWGSize:512  args: 7 teamsXthrds:([[S:[ ]*]][[NUM_TEAMS:[0-9]+]]X 512)
/// CHECK: DEVID:[[S:[ ]*]][[DEVID:[0-9]+]] SGN:8 ConstWGSize:128  args: 7 teamsXthrds:([[S:[ ]*]][[NUM_TEAMS:[0-9]+]]X 128)
/// CHECK: DEVID:[[S:[ ]*]][[DEVID:[0-9]+]] SGN:8 ConstWGSize:128  args: 7 teamsXthrds:([[S:[ ]*]][[NUM_TEAMS:[0-9]+]]X 128)

/// CHECK: DEVID:[[S:[ ]*]][[DEVID:[0-9]+]] SGN:8 ConstWGSize:128  args: 7 teamsXthrds:([[S:[ ]*]][[NUM_TEAMS:[0-9]+]]X 128)
/// CHECK: DEVID:[[S:[ ]*]][[DEVID:[0-9]+]] SGN:8 ConstWGSize:512  args: 7 teamsXthrds:([[S:[ ]*]][[NUM_TEAMS:[0-9]+]]X 512)
/// CHECK: DEVID:[[S:[ ]*]][[DEVID:[0-9]+]] SGN:8 ConstWGSize:128  args: 7 teamsXthrds:([[S:[ ]*]][[NUM_TEAMS:[0-9]+]]X 128)
/// CHECK: DEVID:[[S:[ ]*]][[DEVID:[0-9]+]] SGN:8 ConstWGSize:128  args: 7 teamsXthrds:([[S:[ ]*]][[NUM_TEAMS:[0-9]+]]X 128)

/// CHECK: DEVID:[[S:[ ]*]][[DEVID:[0-9]+]] SGN:8 ConstWGSize:128  args:10 teamsXthrds:([[S:[ ]*]][[NUM_TEAMS:[0-9]+]]X 128)
/// CHECK: DEVID:[[S:[ ]*]][[DEVID:[0-9]+]] SGN:8 ConstWGSize:512  args:10 teamsXthrds:([[S:[ ]*]][[NUM_TEAMS:[0-9]+]]X 512)
/// CHECK: DEVID:[[S:[ ]*]][[DEVID:[0-9]+]] SGN:8 ConstWGSize:128  args:10 teamsXthrds:([[S:[ ]*]][[NUM_TEAMS:[0-9]+]]X 128)
/// CHECK: DEVID:[[S:[ ]*]][[DEVID:[0-9]+]] SGN:8 ConstWGSize:128  args:10 teamsXthrds:([[S:[ ]*]][[NUM_TEAMS:[0-9]+]]X 128)

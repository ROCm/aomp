/*
   Copyright © Advanced Micro Devices, Inc., or its affiliates.

   SPDX-License-Identifier:  MIT
*/
#include <stdio.h>
#include <omp.h>

int main()
{
  int N = 1000;

  double a[N];

  for (int i=0; i<N; i++)
    a[i]=i;

  double sum1, sum2, sum3, sum4;
  sum1 = sum2 = sum3 = sum4 = 0;

#pragma omp target teams map(tofrom:sum1) reduction(+:sum1)
#pragma omp distribute parallel for reduction(+:sum1)
  {
      for (int k = 0; k< N; k++) {
	sum1 += a[k];
      }
  }

#pragma omp target teams map(tofrom:sum1) reduction(+:sum1) thread_limit(64)
#pragma omp distribute parallel for reduction(+:sum1)
  {
      for (int k = 0; k< N; k++) {
	sum1 += a[k];
      }
  }

#pragma omp target map(tofrom:sum1)
#pragma omp teams reduction(+:sum1)
#pragma omp distribute parallel for reduction(+:sum1)
  {
    {
      for (int k = 0; k< N; k++) {
	sum1 += a[k];
      }
    }
  }

#pragma omp target map(tofrom:sum1)
#pragma omp teams reduction(+:sum1)
#pragma omp distribute parallel for reduction(+:sum1) num_threads(128)
  {
    {
      for (int k = 0; k< N; k++) {
	sum1 += a[k];
      }
    }
  }

#pragma omp target map(tofrom:sum1)
#pragma omp teams reduction(+:sum1)
#pragma omp distribute parallel for reduction(+:sum1)
  for (int k = 0; k< N; k++) {
    sum1 += a[k];
  }

#pragma omp target map(tofrom:sum1)
#pragma omp teams reduction(+:sum1)
#pragma omp distribute parallel for reduction(+:sum1) num_threads(128)
  for (int k = 0; k< N; k++) {
    sum1 += a[k];
  }

// The following generates SGN:3 today and generates incorrect results.
// TODO revisit
#if 0  
#pragma omp target map(tofrom:sum1)
#pragma omp teams
  {
#pragma omp distribute parallel for reduction(+:sum1)
    for (int k = 0; k< N; k++)
      sum1 += a[k];
#pragma omp distribute parallel for reduction(+:sum1)
    for (int k = 0; k< N; k++)
      sum1 += a[k];
  }
#endif
  
#pragma omp target map(tofrom:sum1)  
#pragma omp teams distribute parallel for reduction(+:sum1)
  for (int j = 0; j< N; j=j+1)
    sum1 += a[j];
  
#pragma omp target map(tofrom:sum1)  
#pragma omp teams distribute parallel for reduction(+:sum1) thread_limit(512) num_threads(128)
  for (int j = 0; j< N; j=j+1)
    sum1 += a[j];
  
#pragma omp target map(tofrom:sum1)
  {
#pragma omp teams distribute parallel for reduction(+:sum1)
    for (int j = 0; j< N; j=j+1)
      sum1 += a[j];
  }

#pragma omp target map(tofrom:sum1)
  {
#pragma omp teams distribute parallel for reduction(+:sum1) thread_limit(512) num_threads(256)
    for (int j = 0; j< N; j=j+1)
      sum1 += a[j];
  }

#pragma omp target teams distribute parallel for map(tofrom:sum2) reduction(+:sum2)
  for (int j = 0; j< N; j=j+2)
    sum2 += a[j];

#pragma omp target teams distribute parallel for map(tofrom:sum2) reduction(+:sum2) thread_limit(512) num_threads(128)
  for (int j = 0; j< N; j=j+2)
    sum2 += a[j];

#pragma omp target teams distribute parallel for map(tofrom:sum3,sum4) reduction(+:sum3,sum4)
  for (int j = 0; j< N; j=j+1) {
    sum3 += a[j];
    sum4 += a[j];
  }
  
#pragma omp target teams distribute parallel for map(tofrom:sum3,sum4) reduction(+:sum3,sum4) thread_limit(64) num_threads(128)
  for (int j = 0; j< N; j=j+1) {
    sum3 += a[j];
    sum4 += a[j];
  }
  
  printf("%f %f %f %f\n", sum1, sum2, sum3, sum4);
  
  int rc =
    (sum1 != 4995000) ||
    (sum2 != 499000) ||
    (sum3 != 999000) ||
    (sum4 != 999000);

  if (!rc)
    printf("Success\n");

  return rc;
}

// Cross-team reductions are emitted as plain SPMD kernels (SGN:2); the
// downstream Xteam reduction execution mode (SGN:8) has been removed, and with
// it the two extra kernel arguments it used to add ('args: 7'/'args:10' before,
// 'args: 5'/'args: 6' now).
//
// The teams-level 'reduction' clause on the split (non-combined) forms above is
// required: without it every team accumulates into the same shared variable,
// which is a data race. The removed downstream implementation used to paper
// over that by always reducing across teams.
// Every kernel here has a block size of 64: an explicitly requested
// '-fopenmp-target-xteam-reduction-blocksize=' overrides the thread_limit and
// num_threads clauses on the constructs below, which is how the block size of
// a cross-team reduction kernel has always been selected.
/// CHECK: DEVID:[[S:[ ]*]][[DEVID:[0-9]+]] SGN:2 ConstWGSize:64 args: 5 teamsXthrds:([[S:[ ]*]][[NUM_TEAMS:[0-9]+]]X 32)
/// CHECK: DEVID:[[S:[ ]*]][[DEVID:[0-9]+]] SGN:2 ConstWGSize:64 args: 5 teamsXthrds:([[S:[ ]*]][[NUM_TEAMS:[0-9]+]]X 64)
/// CHECK: DEVID:[[S:[ ]*]][[DEVID:[0-9]+]] SGN:2 ConstWGSize:64 args: 5 teamsXthrds:([[S:[ ]*]][[NUM_TEAMS:[0-9]+]]X 32)
/// CHECK: DEVID:[[S:[ ]*]][[DEVID:[0-9]+]] SGN:2 ConstWGSize:64 args: 5 teamsXthrds:([[S:[ ]*]][[NUM_TEAMS:[0-9]+]]X 32)
/// CHECK: DEVID:[[S:[ ]*]][[DEVID:[0-9]+]] SGN:2 ConstWGSize:64 args: 5 teamsXthrds:([[S:[ ]*]][[NUM_TEAMS:[0-9]+]]X 32)
/// CHECK: DEVID:[[S:[ ]*]][[DEVID:[0-9]+]] SGN:2 ConstWGSize:64 args: 5 teamsXthrds:([[S:[ ]*]][[NUM_TEAMS:[0-9]+]]X 32)
/// CHECK: DEVID:[[S:[ ]*]][[DEVID:[0-9]+]] SGN:2 ConstWGSize:64 args: 5 teamsXthrds:([[S:[ ]*]][[NUM_TEAMS:[0-9]+]]X 32)
/// CHECK: DEVID:[[S:[ ]*]][[DEVID:[0-9]+]] SGN:2 ConstWGSize:64 args: 5 teamsXthrds:([[S:[ ]*]][[NUM_TEAMS:[0-9]+]]X 64)
/// CHECK: DEVID:[[S:[ ]*]][[DEVID:[0-9]+]] SGN:2 ConstWGSize:64 args: 5 teamsXthrds:([[S:[ ]*]][[NUM_TEAMS:[0-9]+]]X 32)
/// CHECK: DEVID:[[S:[ ]*]][[DEVID:[0-9]+]] SGN:2 ConstWGSize:64 args: 5 teamsXthrds:([[S:[ ]*]][[NUM_TEAMS:[0-9]+]]X 64)
/// CHECK: DEVID:[[S:[ ]*]][[DEVID:[0-9]+]] SGN:2 ConstWGSize:64 args: 5 teamsXthrds:([[S:[ ]*]][[NUM_TEAMS:[0-9]+]]X 32)
/// CHECK: DEVID:[[S:[ ]*]][[DEVID:[0-9]+]] SGN:2 ConstWGSize:64 args: 5 teamsXthrds:([[S:[ ]*]][[NUM_TEAMS:[0-9]+]]X 64)
/// CHECK: DEVID:[[S:[ ]*]][[DEVID:[0-9]+]] SGN:2 ConstWGSize:64 args: 6 teamsXthrds:([[S:[ ]*]][[NUM_TEAMS:[0-9]+]]X 32)
/// CHECK: DEVID:[[S:[ ]*]][[DEVID:[0-9]+]] SGN:2 ConstWGSize:64 args: 6 teamsXthrds:([[S:[ ]*]][[NUM_TEAMS:[0-9]+]]X 64)

#include <stdio.h>
#include <omp.h>

int main()
{
  int N = 10;

  int a[N];
  int b[N];
  int c[N];

  int i;

  for (i=0; i<N; i++)
    b[i]=i;

  for (i=0; i<N; i++) {
    a[i] = 0;
    c[i] = 0;
  }

#pragma omp target teams thread_limit(64)
#pragma omp distribute parallel for
  {
      for (int k = 0; k< N; k++) {
	a[k]=b[k];
      }
  }

#pragma omp target teams thread_limit(100)
#pragma omp distribute parallel for
  {
      for (int k = 0; k< N; k++) {
	a[k]=b[k];
      }
  }

#pragma omp target 
#pragma omp teams
#pragma omp distribute parallel for num_threads(256)
  for (int k = 0; k< N; k++) {
    c[k]=b[k];
  }

#pragma omp target 
#pragma omp teams
#pragma omp distribute parallel for num_threads(1024)
  for (int k = 0; k< N; k++) {
    c[k]=b[k];
  }

#pragma omp target 
#pragma omp teams
#pragma omp distribute parallel for num_threads(900)
  for (int k = 0; k< N; k++) {
    c[k]=b[k];
  }

#pragma omp target 
#pragma omp teams
  {
#pragma omp distribute parallel for    
    for (int k = 0; k< N; k++)
      a[k]=b[k];
#pragma omp distribute parallel for    
    for (int k = 0; k< N; k+=2)
      c[k]=b[k];
  }
  
#pragma omp target 
#pragma omp teams distribute parallel for thread_limit(512) num_threads(128)
    for (int k = 0; k< N; k++)
      a[k]=b[k];

#pragma omp target 
  {
#pragma omp teams distribute parallel for thread_limit(64) num_threads(512)
    for (int k = 0; k< N; k++)
      a[k]=b[k];
  }

#pragma omp target 
#pragma omp teams distribute parallel for
    for (int k = 0; k< N/2; k+=2)
      a[k]=b[k];

  int rc = 0;
  for (i=0; i<N; i++) {
    if (a[i] != b[i] || c[i] != b[i]) {
      rc++;
      printf ("Wrong value: a[%d]=%d c[%d]=%d\n", i, a[i], i, c[i]);
    }
  }
  if (!rc)
    printf("Success\n");

  return rc;
}

/// CHECK: DEVID:[[S:[ ]*]][[DEVID:[0-9]+]] SGN:4 ConstWGSize:64  args: 6 teamsXthrds:([[S:[ ]*]][[NUM_TEAMS:[0-9]+]]X 64)
/// CHECK: DEVID:[[S:[ ]*]][[DEVID:[0-9]+]] SGN:4 ConstWGSize:100  args: 6 teamsXthrds:([[S:[ ]*]][[NUM_TEAMS:[0-9]+]]X 100)
/// CHECK: DEVID:[[S:[ ]*]][[DEVID:[0-9]+]] SGN:4 ConstWGSize:256  args: 6 teamsXthrds:([[S:[ ]*]][[NUM_TEAMS:[0-9]+]]X   8)
/// CHECK: DEVID:[[S:[ ]*]][[DEVID:[0-9]+]] SGN:4 ConstWGSize:256  args: 6 teamsXthrds:([[S:[ ]*]][[NUM_TEAMS:[0-9]+]]X   8)
/// CHECK: DEVID:[[S:[ ]*]][[DEVID:[0-9]+]] SGN:4 ConstWGSize:256  args: 6 teamsXthrds:([[S:[ ]*]][[NUM_TEAMS:[0-9]+]]X   8)
/// CHECK: DEVID:[[S:[ ]*]][[DEVID:[0-9]+]] SGN:3 ConstWGSize:257  args: 8 teamsXthrds:([[S:[ ]*]][[NUM_TEAMS:[0-9]+]]X 256)
/// CHECK: DEVID:[[S:[ ]*]][[DEVID:[0-9]+]] SGN:4 ConstWGSize:128  args: 6 teamsXthrds:([[S:[ ]*]][[NUM_TEAMS:[0-9]+]]X 128)
/// CHECK: DEVID:[[S:[ ]*]][[DEVID:[0-9]+]] SGN:4 ConstWGSize:64  args: 6 teamsXthrds:([[S:[ ]*]][[NUM_TEAMS:[0-9]+]]X 64)
/// CHECK: DEVID:[[S:[ ]*]][[DEVID:[0-9]+]] SGN:4 ConstWGSize:256  args: 6 teamsXthrds:([[S:[ ]*]][[NUM_TEAMS:[0-9]+]]X   8)


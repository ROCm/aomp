#include <stdio.h>
#include <omp.h>

int main()
{
  int N = 10000;

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

#pragma omp target teams num_teams(5) thread_limit(128)
#pragma omp distribute parallel for
    {
      for (int k = 0; k< N; k++) {
	a[k]=b[k];
      }
  }

#pragma omp target
#pragma omp teams num_teams(20) 
#pragma omp distribute parallel for num_threads(64)
  {
    {
      for (int k = 0; k< N; k++) {
	c[k]=b[k];
      }
    }
  }

#pragma omp target 
#pragma omp teams num_teams(20) thread_limit(768)
#pragma omp distribute parallel for
  for (int k = 0; k< N; k++) {
    c[k]=b[k];
  }

#pragma omp target teams
#pragma omp distribute parallel for
    {
      for (int k = 0; k< N; k++) {
	a[k]=b[k];
      }
  }

#pragma omp target
#pragma omp teams
#pragma omp distribute parallel for
  {
    {
      for (int k = 0; k< N; k++) {
	c[k]=b[k];
      }
    }
  }

#pragma omp target 
#pragma omp teams
#pragma omp distribute parallel for
  for (int k = 0; k< N; k++) {
    c[k]=b[k];
  }

#pragma omp target 
#pragma omp teams num_teams(20) thread_limit(128)
  {
#pragma omp distribute parallel for    
    for (int k = 0; k< N; k++)
      a[k]=b[k];
#pragma omp distribute parallel for    
    for (int k = 0; k< N; k+=2)
      c[k]=b[k];
  }
  
#pragma omp target 
#pragma omp teams distribute parallel for num_teams(20) num_threads(512)
    for (int k = 0; k< N; k++)
      a[k]=b[k];

#pragma omp target 
  {
#pragma omp teams distribute parallel for num_teams(7) num_threads(1024)
    for (int k = 0; k< N; k++)
      a[k]=b[k];
  }

#pragma omp target 
#pragma omp teams distribute parallel for num_teams(20) thread_limit(64)
    for (int k = 0; k< N/2; k+=2)
      a[k]=b[k];

#pragma omp target 
#pragma omp teams distribute parallel for
    for (int k = 0; k< N; k++)
      a[k]=b[k];

#pragma omp target 
  {
#pragma omp teams distribute parallel for
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

/// CHECK: DEVID:[[S:[ ]*]][[DEVID:[0-9]+]] SGN:5 ConstWGSize:1024  args: 6 teamsXthrds:([[S:[ ]*]][[NUM_TEAMS:[0-9]+]]X 128)
/// CHECK: DEVID:[[S:[ ]*]][[DEVID:[0-9]+]] SGN:5 ConstWGSize:1024  args: 6 teamsXthrds:([[S:[ ]*]][[NUM_TEAMS:[0-9]+]]X1024)
/// CHECK: DEVID:[[S:[ ]*]][[DEVID:[0-9]+]] SGN:5 ConstWGSize:1024  args: 6 teamsXthrds:([[S:[ ]*]][[NUM_TEAMS:[0-9]+]]X 768)
/// CHECK: DEVID:[[S:[ ]*]][[DEVID:[0-9]+]] SGN:5 ConstWGSize:1024  args: 6 teamsXthrds:([[S:[ ]*]][[NUM_TEAMS:[0-9]+]]X1024)
/// CHECK: DEVID:[[S:[ ]*]][[DEVID:[0-9]+]] SGN:5 ConstWGSize:1024  args: 6 teamsXthrds:([[S:[ ]*]][[NUM_TEAMS:[0-9]+]]X1024)
/// CHECK: DEVID:[[S:[ ]*]][[DEVID:[0-9]+]] SGN:5 ConstWGSize:1024  args: 6 teamsXthrds:([[S:[ ]*]][[NUM_TEAMS:[0-9]+]]X1024)
/// CHECK: DEVID:[[S:[ ]*]][[DEVID:[0-9]+]] SGN:3 ConstWGSize:257  args: 8 teamsXthrds:([[S:[ ]*]][[NUM_TEAMS:[0-9]+]]X 128)
/// CHECK: DEVID:[[S:[ ]*]][[DEVID:[0-9]+]] SGN:5 ConstWGSize:1024  args: 6 teamsXthrds:([[S:[ ]*]][[NUM_TEAMS:[0-9]+]]X 512)
/// CHECK: DEVID:[[S:[ ]*]][[DEVID:[0-9]+]] SGN:5 ConstWGSize:1024  args: 6 teamsXthrds:([[S:[ ]*]][[NUM_TEAMS:[0-9]+]]X1024)
/// CHECK: DEVID:[[S:[ ]*]][[DEVID:[0-9]+]] SGN:5 ConstWGSize:1024  args: 6 teamsXthrds:([[S:[ ]*]][[NUM_TEAMS:[0-9]+]]X  64)
/// CHECK: DEVID:[[S:[ ]*]][[DEVID:[0-9]+]] SGN:5 ConstWGSize:1024  args: 6 teamsXthrds:([[S:[ ]*]][[NUM_TEAMS:[0-9]+]]X1024)
/// CHECK: DEVID:[[S:[ ]*]][[DEVID:[0-9]+]] SGN:5 ConstWGSize:1024  args: 6 teamsXthrds:([[S:[ ]*]][[NUM_TEAMS:[0-9]+]]X1024)
/// CHECK: DEVID:[[S:[ ]*]][[DEVID:[0-9]+]] SGN:5 ConstWGSize:1024  args: 6 teamsXthrds:([[S:[ ]*]][[NUM_TEAMS:[0-9]+]]X1024)


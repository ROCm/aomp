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

// A kernel with a num_teams or thread_limit clause gets the full block size
// (ConstWGSize, or the thread_limit or num_threads clause) and as many teams as
// its trip count needs, at most num_teams. Where num_teams asks for more teams
// than needed, the runtime may launch either. The other kernels are too short
// for the default number of teams of full blocks, so the low-trip-count rule
// gives them fewer threads per team, down to 32, and more teams. How many
// depends on the default number of teams of the device (four per CU on AMDGPU);
// the alternatives cover every device with at least 16 CUs.
/// CHECK: DEVID:[[S:[ ]*]][[DEVID:[0-9]+]] SGN:7 ConstWGSize:1024  args: 6 teamsXthrds:({{ *}}5X 128)
/// CHECK: DEVID:[[S:[ ]*]][[DEVID:[0-9]+]] SGN:7 ConstWGSize:1024  args: 6 teamsXthrds:({{ *}}{{1[0-9]|20}}X1024)
/// CHECK: DEVID:[[S:[ ]*]][[DEVID:[0-9]+]] SGN:7 ConstWGSize:1024  args: 6 teamsXthrds:({{ *}}{{1[0-9]|20}}X 768)
/// CHECK: DEVID:[[S:[ ]*]][[DEVID:[0-9]+]] SGN:7 ConstWGSize:1024  args: 6 teamsXthrds:({{ *}}{{313X  32|157X  64|79X 128|40X 256}})
/// CHECK: DEVID:[[S:[ ]*]][[DEVID:[0-9]+]] SGN:7 ConstWGSize:1024  args: 6 teamsXthrds:({{ *}}{{313X  32|157X  64|79X 128|40X 256}})
/// CHECK: DEVID:[[S:[ ]*]][[DEVID:[0-9]+]] SGN:7 ConstWGSize:1024  args: 6 teamsXthrds:({{ *}}{{313X  32|157X  64|79X 128|40X 256}})
/// CHECK: DEVID:[[S:[ ]*]][[DEVID:[0-9]+]] SGN:3 ConstWGSize:{{288|320}}  args: 8 teamsXthrds:({{ *}}20X 128)
/// CHECK: DEVID:[[S:[ ]*]][[DEVID:[0-9]+]] SGN:7 ConstWGSize:1024  args: 6 teamsXthrds:({{ *}}20X 512)
/// CHECK: DEVID:[[S:[ ]*]][[DEVID:[0-9]+]] SGN:7 ConstWGSize:1024  args: 6 teamsXthrds:({{ *}}7X1024)
/// CHECK: DEVID:[[S:[ ]*]][[DEVID:[0-9]+]] SGN:7 ConstWGSize:1024  args: 6 teamsXthrds:({{ *}}20X  64)
/// CHECK: DEVID:[[S:[ ]*]][[DEVID:[0-9]+]] SGN:7 ConstWGSize:1024  args: 6 teamsXthrds:({{ *}}{{313X  32|157X  64|79X 128|40X 256}})
/// CHECK: DEVID:[[S:[ ]*]][[DEVID:[0-9]+]] SGN:7 ConstWGSize:1024  args: 6 teamsXthrds:({{ *}}{{313X  32|157X  64|79X 128|40X 256}})
/// CHECK: DEVID:[[S:[ ]*]][[DEVID:[0-9]+]] SGN:7 ConstWGSize:1024  args: 6 teamsXthrds:({{ *}}{{79X  32|40X  64}})


#include <stdio.h>
#include <omp.h>

int main() {
  int num_threads = 0;
  int N = 100000;

  int a[N];
  int b[N];
  int c[N];

  int i;

#pragma omp target map(from: num_threads)
  {
    num_threads = omp_get_num_threads();
  }
  printf("num_threads = %d\n", num_threads);
  
  for (i=0; i<N; i++)
    a[i]=0;

  for (i=0; i<N; i++)
    b[i]=i;

#pragma omp target parallel for
  {
    for (int j = 0; j< N; j++)
      a[j]=b[j];
  }

#pragma omp target teams
  {
#pragma omp distribute parallel for
    for (int j = 0; j< N; j++)
      a[j]=b[j];
#pragma omp distribute parallel for
    for (int j = 0; j< N; j++)
      c[j]=b[j];
  }
  
#pragma omp target teams distribute parallel for
  {
    for (int j = 0; j< N; j++)
      a[j]=b[j];
  }

#pragma omp target teams distribute parallel for thread_limit(64)
  {
    for (int j = 0; j< N; j++)
      a[j]=b[j];
  }

  int rc = 0;
  for (i=0; i<N; i++)
    if (a[i] != b[i] || c[i] != b[i]) {
      rc++;
      printf ("Wrong value: a[%d]=%d c[%d]=%d\n", i, a[i], i, c[i]);
    }

  if (!rc)
    printf("Success\n");

  return rc;
}

// Compiled with default options

/// CHECK: DEVID: 0 SGN:1 ConstWGSize:257  args: 1 teamsXthrds:([[S:[ ]*]][[NUM_TEAMS:[0-9]+]]X 257)
/// CHECK: DEVID: 0 SGN:2 ConstWGSize:256  args: 5 teamsXthrds:([[S:[ ]*]][[NUM_TEAMS:[0-9]+]]X 256)
/// CHECK: DEVID: 0 SGN:3 ConstWGSize:257  args: 7 teamsXthrds:([[S:[ ]*]][[NUM_TEAMS:[0-9]+]]X 256)
/// CHECK: DEVID: 0 SGN:2 ConstWGSize:256  args: 5 teamsXthrds:([[S:[ ]*]][[NUM_TEAMS:[0-9]+]]X 256)
/// CHECK: DEVID: 0 SGN:2 ConstWGSize:256  args: 5 teamsXthrds:([[S:[ ]*]][[NUM_TEAMS:[0-9]+]]X 64)

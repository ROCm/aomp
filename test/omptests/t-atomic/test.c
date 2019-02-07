#include <stdio.h>

#include "../utilities/check.h"

#define N 100

int main()
{
  check_offloading();

  // Initialisation
  int fail = 0;
  int error = 0;
  int a[N];
  int b[N];

  /* 
   * Atomics update (implicit)
   */

  // Initialise
  a[0] = 0;

  // Execute
#pragma omp target map(tofrom: a[:1])
  {
#pragma omp parallel for
    for(int ii = 0; ii < N; ++ii)
#pragma omp atomic
      a[0]++;
  }

  // Check result
  int result = a[0];
  int expect = N;
  if (result != expect) 
  {
    printf("update (implicit) a %d != %d (error %d)\n", 
        result, expect, ++error);
    fail = 1;
  }

  if(!fail)
    printf("successful\n");
  else
    fail = 0;

  /* 
   * Atomics update (explicit)
   */

  // Initialise
  a[0] = 0;

  // Execute
#pragma omp target map(tofrom: a[:1])
  {
#pragma omp parallel for
    for(int ii = 0; ii < N; ++ii)
#pragma omp atomic update
      a[0]++;
  }

  // Check result
  result = a[0];
  expect = N;
  if (result != expect) 
  {
    printf("update (explicit) a %d != %d (error %d)\n", 
        result, expect, ++error);
    fail = 1;
  }

  if(!fail)
    printf("successful\n");
  else
    fail = 0;

  /* 
   * Atomic capture
   */

  // Initilisation
  a[0] = 0;
  for(int ii = 0; ii < N; ++ii)
    b[ii] = -1;

  // Execute
#pragma omp target map(tofrom: a[:1], b[:N])
  {
#pragma omp parallel for
    for(int ii = 0; ii < N; ++ii)
    {
      int v = 0;
#pragma omp atomic capture
      v = a[0]++;

      b[ii] = v;
    }
  }

  // Check result
  result = a[0];
  expect = N;
  if (result != expect) 
  {
    printf("capture a %d != %d (error %d)\n", 
        result, expect, ++error);
    fail = 1;
  }

  // Make sure every increment was captured, regardless of order
  for(int ii = 0; ii < N; ++ii)
  {
    int pass = 0;
    expect = ii;
    for(int jj = 0; jj < N; ++jj)
    {
      result = b[jj];
      if(result == expect)
        pass = 1;
    }
    if (!pass) 
    {
      printf("capture b %d not captured (error %d)\n", 
          expect, ++error);
      fail = 1;
    }
  }

  if(!fail)
    printf("successful\n");
  else
    fail = 0;

  /* 
   * Atomic write
   */

  // Initialisation
  for(int ii = 0; ii < N; ++ii)
    a[ii] = 0;

  // Execute
#pragma omp target map(tofrom: a[:N])
  {
#pragma omp parallel for
    for(int ii = 0; ii < N; ++ii)
#pragma omp atomic write
      a[ii] = ii;
  }

  // Check result
  for(int ii = 0; ii < N; ++ii)
  {
    result = a[ii];
    expect = ii;
    if (result != expect) 
    {
      printf("write a %d != %d (error %d)\n", 
          result, expect, ++error);
      fail = 1;
    }
  }

  if(!fail)
    printf("successful\n");
  else
    fail = 0;

  /* 
   * Atomic read
   */

  // Initialisation
  for(int ii = 0; ii < N; ++ii)
  {
    a[ii] = 0;
    b[ii] = ii;
  }

  // Execute
#pragma omp target map(tofrom: a[:N], b[:N])
  {
#pragma omp parallel for
    for(int ii = 0; ii < N; ++ii)
#pragma omp atomic read
      a[ii] = b[ii];
  }

  // Check result
  for(int ii = 0; ii < N; ++ii)
  {
    result = a[ii];
    expect = b[ii];
    if (result != expect) 
    {
      printf("ar a %d != %d (error %d)\n", 
          result, expect, ++error);
      fail = 1;
    }
  }

  if(!fail)
    printf("successful\n");
  else
    fail = 0;

  /* 
   * Atomics update with multiple teams
   */

  // Initialise
  a[0] = 0;

  // Execute
#pragma omp target map(tofrom: a[:1])
  {
#pragma omp teams num_teams(10) thread_limit(10)
#pragma omp distribute parallel for
    for(int ii = 0; ii < N; ++ii)
#pragma omp atomic
      a[0]++;
  }

  // Check result
  result = a[0];
  expect = N;
  if (result != expect) 
  {
    printf("Multi Team a %d != %d (error %d)\n", 
        result, expect, ++error);
    fail = 1;
  }

  if(!fail)
    printf("successful\n");
  else
    fail = 0;

  /* 
   * Atomics seq_cst
   */

  // Initialise
  a[0] = 0;

  // Execute
#pragma omp target map(tofrom: a[:1])
  {
#pragma omp parallel for
    for(int ii = 0; ii < N; ++ii)
#pragma omp atomic seq_cst
      a[0]++;
  }

  // Check result
  result = a[0];
  expect = N;
  if (result != expect) 
  {
    printf("Using seq_cst a %d != %d (error %d)\n", 
        result, expect, ++error);
    fail = 1;
  }

  if(!fail)
    printf("successful\n");
  else
    fail = 0;

  // Report
  printf("done with %d errors\n", error);
  return error;
}

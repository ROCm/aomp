#include <stdlib.h>
#include <stdio.h>

#include "omp.h"

#include "../utilities/check.h"
#include "../utilities/utilities.h"

#define N 10

int main()
{
  double a[N], a_h[N];
  double b[N], c[N];
  double d[N], d_h[N];
  int fail = 0;

  check_offloading();

  long cpuExec = 0;
#pragma omp target map(tofrom: cpuExec)
  {
    cpuExec = omp_is_initial_device();
  }

  // taskloop is only implemented on the gpu
  if (!cpuExec) {

    // Test: basic with shared

    for(int i = 0 ; i < N ; i++) {
      a[i] = a_h[i] = 0;
      b[i] = i;
      c[i] = i-7;
      d[i] = d_h[i] = i+12;
    }

#pragma omp target map(tofrom:a) map(to:b,c)
    {
#pragma omp parallel
#pragma omp single
#pragma omp taskgroup
#pragma omp taskloop shared(a)
      for(int i = 0 ; i < N; i++) {
	d[i] += b[i] + c[i];
      }
      // handle dependency between two taskloop using taskgroup
      // as tasks are immediately executed, no need for further
      // logic to synchronize
#pragma omp taskgroup
#pragma omp taskloop shared(a)
      for(int i = 0 ; i < N; i++) {
	a[i] += d[i];
      }

    }

    for(int i = 0 ; i < N; i++) {
      d_h[i] += b[i] + c[i];
      a_h[i] += d_h[i];
    }

    for(int i = 0 ; i < N; i++)
      if (a[i] != a_h[i]) {
	printf("Error %d: device = %lf, host = %lf\n", i, a[i], a_h[i]);
	fail = 1;
      }

    if (fail)
      printf("Failed\n");
    else
      printf("Succeeded\n");

  } else // if !cpuExec
    DUMP_SUCCESS(1);

  return 0;
}

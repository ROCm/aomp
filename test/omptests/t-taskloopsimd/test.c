
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
    }

#pragma omp target map(tofrom:a) map(to:b,c)
    {
#pragma omp parallel
#pragma omp single
#pragma omp taskloop simd shared(a)
      for(int i = 0 ; i < N; i++) {
	a[i] += b[i] + c[i];
      }
    }

    for(int i = 0 ; i < N; i++)
      a_h[i] += b[i] + c[i];


    for(int i = 0 ; i < N; i++)
      if (a[i] != a_h[i]) {
	printf("Error %d: device = %lf, host = %lf\n", i, a[i], a_h[i]);
	fail = 1;
      }

    if (fail)
      printf("Failed\n");
    else
      printf("Succeeded\n");

    // Test: if clause
    fail = 0;
    for(int i = 0 ; i < N ; i++) {
      a[i] = a_h[i] = 0;
      b[i] = i;
      c[i] = i-7;
    }

#pragma omp target map(tofrom:a) map(to:b,c)
    {
#pragma omp parallel
#pragma omp single
#pragma omp taskloop simd shared(a) if(0) //undeferred execution of task
      for(int i = 0 ; i < N; i++) {
	a[i] += b[i] + c[i];
      }
    }

    for(int i = 0 ; i < N; i++)
      a_h[i] += b[i] + c[i];


    for(int i = 0 ; i < N; i++)
      if (a[i] != a_h[i]) {
	printf("Error %d: device = %lf, host = %lf\n", i, a[i], a_h[i]);
	fail = 1;
      }

    if (fail)
      printf("Failed\n");
    else
      printf("Succeeded\n");

    // Test: private clause
    fail = 0;
    for(int i = 0 ; i < N ; i++) {
      a[i] = a_h[i] = 0;
      b[i] = i;
      c[i] = i-7;
    }

    int myId = -1;
#pragma omp target map(tofrom:a) map(to:b,c)
    {
#pragma omp parallel
#pragma omp single
#pragma omp taskloop simd shared(a) private(myId)
      for(int i = 0 ; i < N; i++) {
	myId = omp_get_thread_num();
	a[i] += b[i] + c[i] + myId;
      }
    }

    // myId == 0 for all iterations because we execute the entire loop on a single thread (the master)
    for(int i = 0 ; i < N; i++)
      a_h[i] += b[i] + c[i] + 0;


    for(int i = 0 ; i < N; i++)
      if (a[i] != a_h[i]) {
	printf("Error %d: device = %lf, host = %lf\n", i, a[i], a_h[i]);
	fail = 1;
      }

    if (fail)
      printf("Failed\n");
    else
      printf("Succeeded\n");


    // Test: firstprivate clause
    fail = 0;
    for(int i = 0 ; i < N ; i++) {
      a[i] = a_h[i] = 0;
      b[i] = i;
      c[i] = i-7;
    }

    myId = -1;
#pragma omp target map(tofrom:a) map(to:b,c)
    {
#pragma omp parallel
#pragma omp single
#pragma omp taskloop simd shared(a) firstprivate(myId)
      for(int i = 0 ; i < N; i++) {
	myId += omp_get_thread_num();
	a[i] += b[i] + c[i] + myId;
      }
    }

    // myId == 0 for all iterations because we execute the entire loop on a single thread (the master)+
    for(int i = 0 ; i < N; i++)
      a_h[i] += b[i] + c[i] + (-1);


    for(int i = 0 ; i < N; i++)
      if (a[i] != a_h[i]) {
	printf("Error %d: device = %lf, host = %lf\n", i, a[i], a_h[i]);
	fail = 1;
      }

    if (fail)
      printf("Failed\n");
    else
      printf("Succeeded\n");

     // Test: lastprivate clause
    fail = 0;
    for(int i = 0 ; i < N ; i++) {
      a[i] = a_h[i] = 0;
      b[i] = i;
      c[i] = i-7;
    }

    double lp = -1;
#pragma omp target map(tofrom:a) map(to:b,c) map(tofrom:lp)
    {
#pragma omp parallel
#pragma omp single
#pragma omp taskloop simd shared(a) lastprivate(myId)
      for(int i = 0 ; i < N; i++) {
	a[i] += b[i] + c[i];
	lp = a[i];
      }
    }

    for(int i = 0 ; i < N; i++)
      a_h[i] += b[i] + c[i];


    for(int i = 0 ; i < N; i++)
      if (a[i] != a_h[i]) {
	printf("Error %d: device = %lf, host = %lf\n", i, a[i], a_h[i]);
	fail = 1;
      }

    if (lp != a[N-1]) {
      	printf("Latpriv Error device = %lf, host = %lf\n", lp, a_h[N-1]);
	fail = 1;
    }

    if (fail)
      printf("Failed\n");
    else
      printf("Succeeded\n");

     // Test: default clause
    fail = 0;
    for(int i = 0 ; i < N ; i++) {
      a[i] = a_h[i] = 0;
      b[i] = i;
      c[i] = i-7;
    }

#pragma omp target map(tofrom:a) map(to:b,c) map(tofrom:lp)
    {
#pragma omp parallel
#pragma omp single
#pragma omp taskloop simd default(shared)
      for(int i = 0 ; i < N; i++) {
	a[i] += b[i] + c[i];
      }
    }

    for(int i = 0 ; i < N; i++)
      a_h[i] += b[i] + c[i];


    for(int i = 0 ; i < N; i++)
      if (a[i] != a_h[i]) {
	printf("Error %d: device = %lf, host = %lf\n", i, a[i], a_h[i]);
	fail = 1;
      }

    if (fail)
      printf("Failed\n");
    else
      printf("Succeeded\n");

    // Test: grainsize
    fail = 0;
    for(int i = 0 ; i < N ; i++) {
      a[i] = a_h[i] = 0;
      b[i] = i;
      c[i] = i-7;
    }

#pragma omp target map(tofrom:a) map(to:b,c) map(tofrom:lp)
    {
#pragma omp parallel
#pragma omp single
#pragma omp taskloop simd grainsize(3)
      for(int i = 0 ; i < N; i++) {
	a[i] += b[i] + c[i];
      }
    }

    for(int i = 0 ; i < N; i++)
      a_h[i] += b[i] + c[i];


    for(int i = 0 ; i < N; i++)
      if (a[i] != a_h[i]) {
	printf("Error %d: device = %lf, host = %lf\n", i, a[i], a_h[i]);
	fail = 1;
      }

    if (fail)
      printf("Failed\n");
    else
      printf("Succeeded\n");


    // Test: num_tasks clause
    fail = 0;
    for(int i = 0 ; i < N ; i++) {
      a[i] = a_h[i] = 0;
      b[i] = i;
      c[i] = i-7;
    }

#pragma omp target map(tofrom:a) map(to:b,c) map(tofrom:lp)
    {
#pragma omp parallel
#pragma omp single
#pragma omp taskloop simd num_tasks(5)
      for(int i = 0 ; i < N; i++) {
	a[i] += b[i] + c[i];
      }
    }

    for(int i = 0 ; i < N; i++)
      a_h[i] += b[i] + c[i];


    for(int i = 0 ; i < N; i++)
      if (a[i] != a_h[i]) {
	printf("Error %d: device = %lf, host = %lf\n", i, a[i], a_h[i]);
	fail = 1;
      }

    if (fail)
      printf("Failed\n");
    else
      printf("Succeeded\n");

    // Test: collapse clause

    fail = 0;
    int ma[N][N], mb[N][N], mc[N][N];
    for (int i = 0 ; i < N ; i++)
      for (int j = 0 ; j < N ; j++) {
	ma[i][j] = -1;
	mb[i][j] = i;
	mc[i][j] = 2*i;
      }

#pragma omp target map(tofrom: ma) map(to: mb,mc)
#pragma omp parallel
#pragma omp single
#pragma omp taskloop simd collapse(2)
    for (int i = 0 ; i < N ; i++)
      for (int j = 0 ; j < N ; j++)
	ma[i][j] += mb[i][j] + mc[i][j];

    for (int i = 0 ; i < N ; i++)
      for (int j = 0 ; j < N ; j++)
	if (ma[i][j] != (-1 + i + 2*i)) {
	  printf("Error at %d: device = %d, host = %d\n", i, ma[i][j], (-1 + i + 2*i));
	  fail = 1;
	}

    if (fail)
      printf ("Failed\n");
    else
      printf("Succeeded\n");

    // Test: final clause
    fail = 0;
    for(int i = 0 ; i < N ; i++) {
      a[i] = a_h[i] = 0;
      b[i] = i;
      c[i] = i-7;
    }

#pragma omp target map(tofrom:a) map(to:b,c) map(tofrom:lp)
    {
#pragma omp parallel
#pragma omp single
#pragma omp taskloop simd final(1)
      for(int i = 0 ; i < N; i++) {
	a[i] += b[i] + c[i];
      }
    }

    for(int i = 0 ; i < N; i++)
      a_h[i] += b[i] + c[i];


    for(int i = 0 ; i < N; i++)
      if (a[i] != a_h[i]) {
	printf("Error %d: device = %lf, host = %lf\n", i, a[i], a_h[i]);
	fail = 1;
      }

    if (fail)
      printf("Failed\n");
    else
      printf("Succeeded\n");

    // Test: priority clause
    fail = 0;
    for(int i = 0 ; i < N ; i++) {
      a[i] = a_h[i] = 0;
      b[i] = i;
      c[i] = i-7;
    }

#pragma omp target map(tofrom:a) map(to:b,c) map(tofrom:lp)
    {
#pragma omp parallel
#pragma omp single
#pragma omp taskloop simd priority(10)
      for(int i = 0 ; i < N; i++) {
	a[i] += b[i] + c[i];
      }
    }

    for(int i = 0 ; i < N; i++)
      a_h[i] += b[i] + c[i];


    for(int i = 0 ; i < N; i++)
      if (a[i] != a_h[i]) {
	printf("Error %d: device = %lf, host = %lf\n", i, a[i], a_h[i]);
	fail = 1;
      }

    if (fail)
      printf("Failed\n");
    else
      printf("Succeeded\n");

    // Test: untied clause
    fail = 0;
    for(int i = 0 ; i < N ; i++) {
      a[i] = a_h[i] = 0;
      b[i] = i;
      c[i] = i-7;
    }

#pragma omp target map(tofrom:a) map(to:b,c) map(tofrom:lp)
    {
#pragma omp parallel
#pragma omp single
#pragma omp taskloop simd untied
      for(int i = 0 ; i < N; i++) {
	a[i] += b[i] + c[i];
      }
    }

    for(int i = 0 ; i < N; i++)
      a_h[i] += b[i] + c[i];


    for(int i = 0 ; i < N; i++)
      if (a[i] != a_h[i]) {
	printf("Error %d: device = %lf, host = %lf\n", i, a[i], a_h[i]);
	fail = 1;
      }

    if (fail)
      printf("Failed\n");
    else
      printf("Succeeded\n");

    // Test: mergeable clause
    fail = 0;
    for(int i = 0 ; i < N ; i++) {
      a[i] = a_h[i] = 0;
      b[i] = i;
      c[i] = i-7;
    }

#pragma omp target map(tofrom:a) map(to:b,c) map(tofrom:lp)
    {
#pragma omp parallel
#pragma omp single
#pragma omp taskloop simd mergeable
      for(int i = 0 ; i < N; i++) {
	a[i] += b[i] + c[i];
      }
    }

    for(int i = 0 ; i < N; i++)
      a_h[i] += b[i] + c[i];


    for(int i = 0 ; i < N; i++)
      if (a[i] != a_h[i]) {
	printf("Error %d: device = %lf, host = %lf\n", i, a[i], a_h[i]);
	fail = 1;
      }

    if (fail)
      printf("Failed\n");
    else
      printf("Succeeded\n");

    // Test: nogroup clause
    fail = 0;
    for(int i = 0 ; i < N ; i++) {
      a[i] = a_h[i] = 0;
      b[i] = i;
      c[i] = i-7;
    }

#pragma omp target map(tofrom:a) map(to:b,c) map(tofrom:lp)
    {
#pragma omp parallel
#pragma omp single
#pragma omp taskloop simd nogroup
      for(int i = 0 ; i < N; i++) {
	a[i] += b[i] + c[i];
      }
    }

    for(int i = 0 ; i < N; i++)
      a_h[i] += b[i] + c[i];


    for(int i = 0 ; i < N; i++)
      if (a[i] != a_h[i]) {
	printf("Error %d: device = %lf, host = %lf\n", i, a[i], a_h[i]);
	fail = 1;
      }

    if (fail)
      printf("Failed\n");
    else
      printf("Succeeded\n");

    // Test: safelen clause
    fail = 0;
    for(int i = 0 ; i < N ; i++) {
      a[i] = a_h[i] = 0;
      b[i] = i;
      c[i] = i-7;
    }

#pragma omp target map(tofrom:a) map(to:b,c) map(tofrom:lp)
    {
#pragma omp parallel
#pragma omp single
#pragma omp taskloop simd safelen(2) // taskloop is sequentialized: safelen can be 0
      for(int i = 0 ; i < N; i++) {
	a[i] += b[i] + c[i];
      }
    }

    for(int i = 0 ; i < N; i++)
      a_h[i] += b[i] + c[i];


    for(int i = 0 ; i < N; i++)
      if (a[i] != a_h[i]) {
	printf("Error %d: device = %lf, host = %lf\n", i, a[i], a_h[i]);
	fail = 1;
      }

    if (fail)
      printf("Failed\n");
    else
      printf("Succeeded\n");

    // Test: simdlen clause
    fail = 0;
    for(int i = 0 ; i < N ; i++) {
      a[i] = a_h[i] = 0;
      b[i] = i;
      c[i] = i-7;
    }

#pragma omp target map(tofrom:a) map(to:b,c) map(tofrom:lp)
    {
#pragma omp parallel
#pragma omp single
#pragma omp taskloop simd simdlen(32)
      for(int i = 0 ; i < N; i++) {
	a[i] += b[i] + c[i];
      }
    }

    for(int i = 0 ; i < N; i++)
      a_h[i] += b[i] + c[i];


    for(int i = 0 ; i < N; i++)
      if (a[i] != a_h[i]) {
	printf("Error %d: device = %lf, host = %lf\n", i, a[i], a_h[i]);
	fail = 1;
      }

    if (fail)
      printf("Failed\n");
    else
      printf("Succeeded\n");

    // compiler assert
#if 0
    // Test: linear clause
    fail = 0;
    for(int i = 0 ; i < N ; i++) {
      a[i] = a_h[i] = 0;
      b[i] = i;
      c[i] = i-7;
    }

    int l = 0;
#pragma omp target map(tofrom:a) map(to:b,c) map(tofrom:lp)
    {
#pragma omp parallel
#pragma omp single
#pragma omp taskloop simd linear(l:2)
      for(int i = 0 ; i < N; i++) {
	a[i] += b[i] + c[i] + l;
      }
    }

    for(int i = 0 ; i < N; i++)
      a_h[i] += b[i] + c[i] + (i*2);


    for(int i = 0 ; i < N; i++)
      if (a[i] != a_h[i]) {
	printf("Error %d: device = %lf, host = %lf\n", i, a[i], a_h[i]);
	fail = 1;
      }

    if (fail)
      printf("Failed\n");
    else
      printf("Succeeded\n");
#endif

    // Test: safelen clause
    fail = 0;
    for(int i = 0 ; i < N ; i++) {
      a[i] = a_h[i] = 0;
      b[i] = i;
      c[i] = i-7;
    }

#pragma omp target map(tofrom:a) map(to:b,c) map(tofrom:lp)
    {
#pragma omp parallel
#pragma omp single
#pragma omp taskloop simd aligned(a,b,c)
      for(int i = 0 ; i < N; i++) {
	a[i] += b[i] + c[i];
      }
    }

    for(int i = 0 ; i < N; i++)
      a_h[i] += b[i] + c[i];


    for(int i = 0 ; i < N; i++)
      if (a[i] != a_h[i]) {
	printf("Error %d: device = %lf, host = %lf\n", i, a[i], a_h[i]);
	fail = 1;
      }

    if (fail)
      printf("Failed\n");
    else
      printf("Succeeded\n");

  }  else // if !cpuExec
    DUMP_SUCCESS(17);

  return 0;
}

#include <stdlib.h>
#include <stdio.h>

#include "../utilities/check.h"
#include "../utilities/utilities.h"

#define N 100

#define TEST_BROKEN 0 /* disable tests reardless of value below */

#define TEST1  1
#define TEST1  1
#define TEST2  1
#define TEST3  1
#define TEST4  1
#define TEST5  1
#define TEST6  1
#define TEST7  1
#define TEST8  1
#define TEST9  1
#define TEST10 1
#define TEST11 1
#define TEST11 1
#define TEST12 1
#define TEST13 1
#define TEST14 1
#define TEST15 1
#define TEST16 1
#define TEST17 1
#define TEST18 1
#define TEST19 1
#define TEST20 1
#define TEST21 1
#define TEST21 1
#define TEST22 1
#define TEST23 1
#define TEST24 1
#define TEST25 1
#define TEST26 1
#define TEST27 1
#define TEST28 1
#define TEST39 1
#define TEST31 1
#define TEST31 1
#define TEST32 1
#define TEST33 1
#define TEST34 1
#define TEST35 1
#define TEST36 1
#define TEST37 1

int main ()
{
  int a[N], b[N], c[N];

#if TEST1
  check_offloading();
#endif


  long cpuExec = 0;
  int fail, ch;
#pragma omp target map(tofrom: cpuExec)
  {
    cpuExec = omp_is_initial_device();
  }

  
#if TEST2
    // Test: no clauses
    fail = 0;
    for (int i = 0 ; i < N ; i++) {
      a[i] = -1;
      b[i] = i;
      c[i] = 2*i;
    }
    if (!cpuExec) {
#pragma omp target map(tofrom: a) map(to: b,c)
#pragma omp parallel
#pragma omp for simd
    for (int i = 0 ; i < N ; i++)
      a[i] += b[i] + c[i];

    for (int i = 0 ; i < N ; i++)
      if (a[i] != (-1 + i + 2*i)) {
	printf("Error at %d: device = %d, host = %d\n", i, a[i], (-1 + i + 2*i));
	fail = 1;
      }
    }
    if (fail)
      printf ("Failed 2\n");
    else
      printf("Succeeded 2\n");
#endif

#if TEST3
    // Test: private, firstprivate, lastprivate, linear
    fail = 0;
    for (int i = 0 ; i < N ; i++) {
      a[i] = -1;
      b[i] = i;
      c[i] = 2*i;
    }
    int q = -5;
    int p = -3;
    int r = 0;
    int l = 10;
    if (!cpuExec) {
#pragma omp target teams num_teams(1) thread_limit(1024) map(tofrom: a, r) map(to: b,c)
#pragma omp parallel
#pragma omp for simd private(q) firstprivate(p) lastprivate(r) linear(l:2)
    for (int i = 0 ; i < N ; i++) {
      q = i + 5;
      p += i + 2;
      a[i] += p*b[i] + c[i]*q +l;
      r = i;
    }
    for (int i = 0 ; i < N ; i++) {
      int expected = (-1 + (-3 + i + 2)*i + (2*i)*(i + 5) + 10+(2*i));
      if (a[i] != expected) {
	      printf("Error at %d: device = %d, host = %d\n", i, a[i], expected);
	      fail = 1;
      }
    }
    if (r != N-1) {
      printf("Error for lastprivate: device = %d, host = %d\n", r, N-1);
      fail = 1;
    }
    }

    if (fail)
      printf ("Failed 3\n");
    else
      printf("Succeeded 3\n");
#endif

#if TEST4
    // Test: schedule static no chunk
    fail = 0;
    for (int i = 0 ; i < N ; i++) {
      a[i] = -1;
      b[i] = i;
      c[i] = 2*i;
    }

    if (!cpuExec) {
#pragma omp target map(tofrom: a) map(to: b,c)
#pragma omp parallel
#pragma omp for simd schedule(static)
    for (int i = 0 ; i < N ; i++)
      a[i] += b[i] + c[i];

    for (int i = 0 ; i < N ; i++)
      if (a[i] != (-1 + i + 2*i)) {
	printf("Error at %d: device = %d, host = %d\n", i, a[i], (-1 + i + 2*i));
	fail = 1;
      }
    }

    if (fail)
      printf ("Failed 4\n");
    else
      printf("Succeeded 4\n");
#endif

#if TEST5
    // Test: schedule static no chunk, monotonic
    fail = 0;
    for (int i = 0 ; i < N ; i++) {
      a[i] = -1;
      b[i] = i;
      c[i] = 2*i;
    }

    if (!cpuExec) {
#pragma omp target teams num_teams(1) thread_limit(1024) map(tofrom: a) map(to: b,c)
#pragma omp parallel
#pragma omp for simd schedule(monotonic: static)
    for (int i = 0 ; i < N ; i++)
      a[i] += b[i] + c[i];

    for (int i = 0 ; i < N ; i++)
      if (a[i] != (-1 + i + 2*i)) {
	printf("Error at %d: device = %d, host = %d\n", i, a[i], (-1 + i + 2*i));
	fail = 1;
      }
    }

    if (fail)
      printf ("Failed 5\n");
    else
      printf("Succeeded 5\n");  
#endif

#if TEST6
    // Test: schedule static no chunk, simd
    fail = 0;
    for (int i = 0 ; i < N ; i++) {
      a[i] = -1;
      b[i] = i;
      c[i] = 2*i;
    }

    if (!cpuExec) {
#pragma omp target map(tofrom: a) map(to: b,c)
#pragma omp parallel
#pragma omp for simd schedule(simd: static)
    for (int i = 0 ; i < N ; i++)
      a[i] += b[i] + c[i];

    for (int i = 0 ; i < N ; i++)
      if (a[i] != (-1 + i + 2*i)) {
	printf("Error at %d: device = %d, host = %d\n", i, a[i], (-1 + i + 2*i));
	fail = 1;
      }
    }

    if (fail)
      printf ("Failed 6\n");
    else
      printf("Succeeded 6\n");  
#endif

#if TEST7
    // Test: schedule static chunk
    fail = 0;
    for (int i = 0 ; i < N ; i++) {
      a[i] = -1;
      b[i] = i;
      c[i] = 2*i;
    }

    ch = 10;
    if (!cpuExec) {
#pragma omp target map(tofrom: a) map(to: b,c)
#pragma omp parallel
#pragma omp for simd schedule(static, ch)
    for (int i = 0 ; i < N ; i++)
      a[i] += b[i] + c[i];

    for (int i = 0 ; i < N ; i++)
      if (a[i] != (-1 + i + 2*i)) {
	printf("Error at %d: device = %d, host = %d\n", i, a[i], (-1 + i + 2*i));
	fail = 1;
      }
    }

    if (fail)
      printf ("Failed 7\n");
    else
      printf("Succeeded 7\n");
#endif

#if TEST8
    // Test: schedule static chunk, monotonic
    fail = 0;
    for (int i = 0 ; i < N ; i++) {
      a[i] = -1;
      b[i] = i;
      c[i] = 2*i;
    }

    ch = 10;
    if (!cpuExec) {
#pragma omp target map(tofrom: a) map(to: b,c)
#pragma omp parallel
#pragma omp for simd schedule(monotonic: static, ch)
    for (int i = 0 ; i < N ; i++)
      a[i] += b[i] + c[i];

    for (int i = 0 ; i < N ; i++)
      if (a[i] != (-1 + i + 2*i)) {
	printf("Error at %d: device = %d, host = %d\n", i, a[i], (-1 + i + 2*i));
	fail = 1;
      }
    }

    if (fail)
      printf ("Failed 8\n");
    else
      printf("Succeeded 8\n");
#endif

#if TEST9
    // Test: schedule static chunk, simd
    fail = 0;
    for (int i = 0 ; i < N ; i++) {
      a[i] = -1;
      b[i] = i;
      c[i] = 2*i;
    }

    ch = 10;
    if (!cpuExec) {
#pragma omp target map(tofrom: a) map(to: b,c)
#pragma omp parallel
#pragma omp for simd schedule(simd: static, ch)
    for (int i = 0 ; i < N ; i++)
      a[i] += b[i] + c[i];

    for (int i = 0 ; i < N ; i++)
      if (a[i] != (-1 + i + 2*i)) {
	printf("Error at %d: device = %d, host = %d\n", i, a[i], (-1 + i + 2*i));
	fail = 1;
      }
    }

    if (fail)
      printf ("Failed 9\n");
    else
      printf("Succeeded 9\n");
#endif

#if TEST10
    // Test: schedule dyanmic no chunk
    fail = 0;
    for (int i = 0 ; i < N ; i++) {
      a[i] = -1;
      b[i] = i;
      c[i] = 2*i;
    }
 
    if (!cpuExec) {
#pragma omp target map(tofrom: a) map(to: b,c)
#pragma omp parallel
#pragma omp for simd schedule(dynamic)
    for (int i = 0 ; i < N ; i++)
      a[i] += b[i] + c[i];

    for (int i = 0 ; i < N ; i++)
      if (a[i] != (-1 + i + 2*i)) {
	printf("Error at %d: device = %d, host = %d\n", i, a[i], (-1 + i + 2*i));
	fail = 1;
      }
    }

    if (fail)
      printf ("Failed 10\n");
    else
      printf("Succeeded 10\n");
#endif

#if TEST11 && TEST_BROKEN // hangs
    // Test: schedule dyanmic no chunk, monotonic
    fail = 0;
    for (int i = 0 ; i < N ; i++) {
      a[i] = -1;
      b[i] = i;
      c[i] = 2*i;
    }
 
    if (!cpuExec) {
#pragma omp target map(tofrom: a) map(to: b,c)
#pragma omp parallel
#pragma omp for simd schedule(monotonic: dynamic)
    for (int i = 0 ; i < N ; i++)
      a[i] += b[i] + c[i];

    for (int i = 0 ; i < N ; i++)
      if (a[i] != (-1 + i + 2*i)) {
	printf("Error at %d: device = %d, host = %d\n", i, a[i], (-1 + i + 2*i));
	fail = 1;
      }
    }

    if (fail)
      printf ("Failed 11\n");
    else
      printf("Succeeded 11\n");
#endif

#if TEST12 && TEST_BROKEN // hangs
    // Test: schedule dyanmic no chunk, nonmonotonic
    fail = 0;
    for (int i = 0 ; i < N ; i++) {
      a[i] = -1;
      b[i] = i;
      c[i] = 2*i;
    }
 
    if (!cpuExec) {
#pragma omp target map(tofrom: a) map(to: b,c)
#pragma omp parallel
#pragma omp for simd schedule(nonmonotonic: dynamic)
    for (int i = 0 ; i < N ; i++)
      a[i] += b[i] + c[i];

    for (int i = 0 ; i < N ; i++)
      if (a[i] != (-1 + i + 2*i)) {
	printf("Error at %d: device = %d, host = %d\n", i, a[i], (-1 + i + 2*i));
	fail = 1;
      }
    }

    if (fail)
      printf ("Failed 12\n");
    else
      printf("Succeeded 12\n");
#endif

#if TEST13
    // Test: schedule dyanmic no chunk, simd
    fail = 0;
    for (int i = 0 ; i < N ; i++) {
      a[i] = -1;
      b[i] = i;
      c[i] = 2*i;
    }
 
    if (!cpuExec) {
#pragma omp target map(tofrom: a) map(to: b,c)
#pragma omp parallel
#pragma omp for simd schedule(simd: dynamic)
    for (int i = 0 ; i < N ; i++)
      a[i] += b[i] + c[i];

    for (int i = 0 ; i < N ; i++)
      if (a[i] != (-1 + i + 2*i)) {
	printf("Error at %d: device = %d, host = %d\n", i, a[i], (-1 + i + 2*i));
	fail = 1;
      }
    }

    if (fail)
      printf ("Failed 13\n");
    else
      printf("Succeeded 13\n");
#endif

#if TEST14
    // Test: schedule dynamic chunk
    fail = 0;
    for (int i = 0 ; i < N ; i++) {
      a[i] = -1;
      b[i] = i;
      c[i] = 2*i;
    }

    ch = 10;
    if (!cpuExec) {
#pragma omp target map(tofrom: a) map(to: b,c)
#pragma omp parallel
#pragma omp for simd schedule(dynamic, ch)
    for (int i = 0 ; i < N ; i++)
      a[i] += b[i] + c[i];

    for (int i = 0 ; i < N ; i++)
      if (a[i] != (-1 + i + 2*i)) {
	printf("Error at %d: device = %d, host = %d\n", i, a[i], (-1 + i + 2*i));
	fail = 1;
      }
    }

    if (fail)
      printf ("Failed 14\n");
    else
      printf("Succeeded 14\n");
#endif

#if TEST15 && TEST_BROKEN
    // Test: schedule dynamic chunk, monotonic
    fail = 0;
    for (int i = 0 ; i < N ; i++) {
      a[i] = -1;
      b[i] = i;
      c[i] = 2*i;
    }

    ch = 10;
    if (!cpuExec) {
#pragma omp target map(tofrom: a) map(to: b,c)
#pragma omp parallel
#pragma omp for simd schedule(monotonic: dynamic, ch)
    for (int i = 0 ; i < N ; i++)
      a[i] += b[i] + c[i];

    for (int i = 0 ; i < N ; i++)
      if (a[i] != (-1 + i + 2*i)) {
	printf("Error at %d: device = %d, host = %d\n", i, a[i], (-1 + i + 2*i));
	fail = 1;
      }
    }

    if (fail)
      printf ("Failed 15\n");
    else
      printf("Succeeded 15\n");
#endif

#if TEST16 && TEST_BROKEN
    // Test: schedule dynamic chunk, nonmonotonic
    fail = 0;
    for (int i = 0 ; i < N ; i++) {
      a[i] = -1;
      b[i] = i;
      c[i] = 2*i;
    }

    ch = 10;
    if (!cpuExec) {
#pragma omp target map(tofrom: a) map(to: b,c)
#pragma omp parallel
#pragma omp for simd schedule(nonmonotonic: dynamic, ch)
    for (int i = 0 ; i < N ; i++)
      a[i] += b[i] + c[i];

    for (int i = 0 ; i < N ; i++)
      if (a[i] != (-1 + i + 2*i)) {
	printf("Error at %d: device = %d, host = %d\n", i, a[i], (-1 + i + 2*i));
	fail = 1;
      }
    }

    if (fail)
      printf ("Failed 16\n");
    else
      printf("Succeeded 16\n");
#endif

#if TEST17
    // Test: schedule dynamic chunk, simd
    fail = 0;
    for (int i = 0 ; i < N ; i++) {
      a[i] = -1;
      b[i] = i;
      c[i] = 2*i;
    }

    ch = 10;
    if (!cpuExec) {
#pragma omp target map(tofrom: a) map(to: b,c)
#pragma omp parallel
#pragma omp for simd schedule(simd: dynamic, ch)
    for (int i = 0 ; i < N ; i++)
      a[i] += b[i] + c[i];

    for (int i = 0 ; i < N ; i++)
      if (a[i] != (-1 + i + 2*i)) {
	printf("Error at %d: device = %d, host = %d\n", i, a[i], (-1 + i + 2*i));
	fail = 1;
      }
    }

    if (fail)
      printf ("Failed 17\n");
    else
      printf("Succeeded 17\n");
#endif

#if TEST18
    // Test: schedule guided no chunk
    fail = 0;
    for (int i = 0 ; i < N ; i++) {
      a[i] = -1;
      b[i] = i;
      c[i] = 2*i;
    }

    if (!cpuExec) {
#pragma omp target map(tofrom: a) map(to: b,c)
#pragma omp parallel
#pragma omp for simd schedule(guided)
    for (int i = 0 ; i < N ; i++)
      a[i] += b[i] + c[i];

    for (int i = 0 ; i < N ; i++)
      if (a[i] != (-1 + i + 2*i)) {
	printf("Error at %d: device = %d, host = %d\n", i, a[i], (-1 + i + 2*i));
	fail = 1;
      }
    }

    if (fail)
      printf ("Failed 18\n");
    else
      printf("Succeeded 18\n");
#endif

#if TEST19 && TEST_BROKEN
    // Test: schedule guided no chunk, monotonic
    fail = 0;
    for (int i = 0 ; i < N ; i++) {
      a[i] = -1;
      b[i] = i;
      c[i] = 2*i;
    }

    if (!cpuExec) {
#pragma omp target map(tofrom: a) map(to: b,c)
#pragma omp parallel
#pragma omp for simd schedule(monotonic: guided)
    for (int i = 0 ; i < N ; i++)
      a[i] += b[i] + c[i];

    for (int i = 0 ; i < N ; i++)
      if (a[i] != (-1 + i + 2*i)) {
	printf("Error at %d: device = %d, host = %d\n", i, a[i], (-1 + i + 2*i));
	fail = 1;
      }
    }

    if (fail)
      printf ("Failed 19\n");
    else
      printf("Succeeded 19\n");
#endif

#if TEST20 && TEST_BROKEN
    // Test: schedule guided no chunk, nonmonotonic
    fail = 0;
    for (int i = 0 ; i < N ; i++) {
      a[i] = -1;
      b[i] = i;
      c[i] = 2*i;
    }

    if (!cpuExec) {
#pragma omp target map(tofrom: a) map(to: b,c)
#pragma omp parallel
#pragma omp for simd schedule(nonmonotonic: guided)
    for (int i = 0 ; i < N ; i++)
      a[i] += b[i] + c[i];

    for (int i = 0 ; i < N ; i++)
      if (a[i] != (-1 + i + 2*i)) {
	printf("Error at %d: device = %d, host = %d\n", i, a[i], (-1 + i + 2*i));
	fail = 1;
      }
    }

    if (fail)
      printf ("Failed 20\n");
    else
      printf("Succeeded 20\n");
#endif

#if TEST21
    // Test: schedule guided no chunk, simd
    fail = 0;
    for (int i = 0 ; i < N ; i++) {
      a[i] = -1;
      b[i] = i;
      c[i] = 2*i;
    }

    if (!cpuExec) {
#pragma omp target map(tofrom: a) map(to: b,c)
#pragma omp parallel
#pragma omp for simd schedule(simd: guided)
    for (int i = 0 ; i < N ; i++)
      a[i] += b[i] + c[i];

    for (int i = 0 ; i < N ; i++)
      if (a[i] != (-1 + i + 2*i)) {
	printf("Error at %d: device = %d, host = %d\n", i, a[i], (-1 + i + 2*i));
	fail = 1;
      }
    }

    if (fail)
      printf ("Failed 21\n");
    else
      printf("Succeeded 21\n");
#endif

#if TEST22
    // Test: schedule guided chunk
    fail = 0;
    for (int i = 0 ; i < N ; i++) {
      a[i] = -1;
      b[i] = i;
      c[i] = 2*i;
    }

    ch = 10;
    if (!cpuExec) {
#pragma omp target map(tofrom: a) map(to: b,c)
#pragma omp parallel
#pragma omp for simd schedule(guided, ch)
    for (int i = 0 ; i < N ; i++)
      a[i] += b[i] + c[i];

    for (int i = 0 ; i < N ; i++)
      if (a[i] != (-1 + i + 2*i)) {
	printf("Error at %d: device = %d, host = %d\n", i, a[i], (-1 + i + 2*i));
	fail = 1;
      }
    }

    if (fail)
      printf ("Failed 22\n");
    else
      printf("Succeeded 22\n");
#endif

#if TEST23 && TEST_BROKEN
    // Test: schedule guided chunk, monotonic
    fail = 0;
    for (int i = 0 ; i < N ; i++) {
      a[i] = -1;
      b[i] = i;
      c[i] = 2*i;
    }

    ch = 10;
    if (!cpuExec) {
#pragma omp target map(tofrom: a) map(to: b,c)
#pragma omp parallel
#pragma omp for simd schedule(monotonic: guided, ch)
    for (int i = 0 ; i < N ; i++)
      a[i] += b[i] + c[i];

    for (int i = 0 ; i < N ; i++)
      if (a[i] != (-1 + i + 2*i)) {
	printf("Error at %d: device = %d, host = %d\n", i, a[i], (-1 + i + 2*i));
	fail = 1;
      }
    }

    if (fail)
      printf ("Failed 23\n");
    else
      printf("Succeeded 23\n");
#endif

#if TEST24 && TEST_BROKEN
    // Test: schedule guided chunk, nonmonotonic
    fail = 0;
    for (int i = 0 ; i < N ; i++) {
      a[i] = -1;
      b[i] = i;
      c[i] = 2*i;
    }

    ch = 10;
    if (!cpuExec) {
#pragma omp target map(tofrom: a) map(to: b,c)
#pragma omp parallel
#pragma omp for simd schedule(nonmonotonic: guided, ch)
    for (int i = 0 ; i < N ; i++)
      a[i] += b[i] + c[i];

    for (int i = 0 ; i < N ; i++)
      if (a[i] != (-1 + i + 2*i)) {
	printf("Error at %d: device = %d, host = %d\n", i, a[i], (-1 + i + 2*i));
	fail = 1;
      }
    }

    if (fail)
      printf ("Failed 24\n");
    else
      printf("Succeeded 24\n");
#endif

#if TEST25
    // Test: schedule guided chunk, simd
    fail = 0;
    for (int i = 0 ; i < N ; i++) {
      a[i] = -1;
      b[i] = i;
      c[i] = 2*i;
    }

    ch = 10;
    if (!cpuExec) {
#pragma omp target map(tofrom: a) map(to: b,c)
#pragma omp parallel
#pragma omp for simd schedule(simd: guided, ch)
    for (int i = 0 ; i < N ; i++)
      a[i] += b[i] + c[i];

    for (int i = 0 ; i < N ; i++)
      if (a[i] != (-1 + i + 2*i)) {
	printf("Error at %d: device = %d, host = %d\n", i, a[i], (-1 + i + 2*i));
	fail = 1;
      }
    }

    if (fail)
      printf ("Failed 25\n");
    else
      printf("Succeeded 25\n");
#endif

#if TEST26
    // Test: schedule auto
    fail = 0;
    for (int i = 0 ; i < N ; i++) {
      a[i] = -1;
      b[i] = i;
      c[i] = 2*i;
    }

    if (!cpuExec) {
#pragma omp target map(tofrom: a) map(to: b,c)
#pragma omp parallel
#pragma omp for simd schedule(auto)
    for (int i = 0 ; i < N ; i++)
      a[i] += b[i] + c[i];

    for (int i = 0 ; i < N ; i++)
      if (a[i] != (-1 + i + 2*i)) {
	printf("Error at %d: device = %d, host = %d\n", i, a[i], (-1 + i + 2*i));
	fail = 1;
      }
    }

    if (fail)
      printf ("Failed 26\n");
    else
      printf("Succeeded 26\n");
#endif

#if TEST27
    // Test: schedule auto, monotonic
    fail = 0;
    for (int i = 0 ; i < N ; i++) {
      a[i] = -1;
      b[i] = i;
      c[i] = 2*i;
    }

    if (!cpuExec) {
#pragma omp target map(tofrom: a) map(to: b,c)
#pragma omp parallel
#pragma omp for simd schedule(monotonic: auto)
    for (int i = 0 ; i < N ; i++)
      a[i] += b[i] + c[i];

    for (int i = 0 ; i < N ; i++)
      if (a[i] != (-1 + i + 2*i)) {
	printf("Error at %d: device = %d, host = %d\n", i, a[i], (-1 + i + 2*i));
	fail = 1;
      }
    }

    if (fail)
      printf ("Failed 27\n");
    else
      printf("Succeeded 27\n");
#endif

#if TEST28
    // Test: schedule auto, simd
    fail = 0;
    for (int i = 0 ; i < N ; i++) {
      a[i] = -1;
      b[i] = i;
      c[i] = 2*i;
    }
    if (!cpuExec) {
#pragma omp target map(tofrom: a) map(to: b,c)
#pragma omp parallel
#pragma omp for simd schedule(simd: auto)
    for (int i = 0 ; i < N ; i++)
      a[i] += b[i] + c[i];

    for (int i = 0 ; i < N ; i++)
      if (a[i] != (-1 + i + 2*i)) {
	printf("Error at %d: device = %d, host = %d\n", i, a[i], (-1 + i + 2*i));
	fail = 1;
      }
    }

    if (fail)
      printf ("Failed 28\n");
    else
      printf("Succeeded 28\n");
#endif

#if TEST29
    // Test: schedule runtime
    fail = 0;
    for (int i = 0 ; i < N ; i++) {
      a[i] = -1;
      b[i] = i;
      c[i] = 2*i;
    }

    if (!cpuExec) {
#pragma omp target map(tofrom: a) map(to: b,c)
#pragma omp parallel
#pragma omp for simd schedule(runtime)
    for (int i = 0 ; i < N ; i++)
      a[i] += b[i] + c[i];

    for (int i = 0 ; i < N ; i++)
      if (a[i] != (-1 + i + 2*i)) {
	printf("Error at %d: device = %d, host = %d\n", i, a[i], (-1 + i + 2*i));
	fail = 1;
      }
    }

    if (fail)
      printf ("Failed 29\n");
    else
      printf("Succeeded 29\n");
#endif

#if TEST30 && TEST_BROKEN
    // Test: schedule runtime, monotonic
    fail = 0;
    for (int i = 0 ; i < N ; i++) {
      a[i] = -1;
      b[i] = i;
      c[i] = 2*i;
    }

    if (!cpuExec) {
#pragma omp target map(tofrom: a) map(to: b,c)
#pragma omp parallel
#pragma omp for simd schedule(monotonic: runtime)
    for (int i = 0 ; i < N ; i++)
      a[i] += b[i] + c[i];

    for (int i = 0 ; i < N ; i++)
      if (a[i] != (-1 + i + 2*i)) {
	printf("Error at %d: device = %d, host = %d\n", i, a[i], (-1 + i + 2*i));
	fail = 1;
      }
    }

    if (fail)
      printf ("Failed 30\n");
    else
      printf("Succeeded 30\n");
#endif

#if TEST31
    // Test: schedule runtime, simd
    fail = 0;
    for (int i = 0 ; i < N ; i++) {
      a[i] = -1;
      b[i] = i;
      c[i] = 2*i;
    }

    if (!cpuExec) {
#pragma omp target map(tofrom: a) map(to: b,c)
#pragma omp parallel
#pragma omp for simd schedule(simd: runtime)
    for (int i = 0 ; i < N ; i++)
      a[i] += b[i] + c[i];

    for (int i = 0 ; i < N ; i++)
      if (a[i] != (-1 + i + 2*i)) {
	printf("Error at %d: device = %d, host = %d\n", i, a[i], (-1 + i + 2*i));
	fail = 1;
      }
    }

    if (fail)
      printf ("Failed 31\n");
    else
      printf("Succeeded 31\n");
#endif

#if TEST32
    // Test: collapse
    fail = 0;
    int ma[N][N], mb[N][N], mc[N][N];
    for (int i = 0 ; i < N ; i++)
      for (int j = 0 ; j < N ; j++) {
	ma[i][j] = -1;
	mb[i][j] = i;
	mc[i][j] = 2*i;
      }

    if (!cpuExec) {
#pragma omp target map(tofrom: ma) map(to: mb,mc)
#pragma omp parallel
#pragma omp for simd collapse(2)
    for (int i = 0 ; i < N ; i++)
      for (int j = 0 ; j < N ; j++)
	ma[i][j] += mb[i][j] + mc[i][j];

    for (int i = 0 ; i < N ; i++)
      for (int j = 0 ; j < N ; j++)
	if (ma[i][j] != (-1 + i + 2*i)) {
	  printf("Error at %d: device = %d, host = %d\n", i, ma[i][j], (-1 + i + 2*i));
	  fail = 1;
	}
    }

    if (fail)
      printf ("Failed 32\n");
    else
      printf("Succeeded 32\n");
#endif

#if TEST33
    // Test: ordered
    fail = 0;
    for (int i = 0 ; i < N ; i++) {
      a[i] = -1;
      b[i] = i;
      c[i] = 2*i;
    }

    if (!cpuExec) {
#pragma omp target map(tofrom: a) map(to: b,c)
#pragma omp parallel
#pragma omp for simd ordered
    for (int i = 0 ; i < N ; i++)
      a[i] += b[i] + c[i];

    for (int i = 0 ; i < N ; i++)
      if (a[i] != (-1 + i + 2*i)) {
	printf("Error at %d: device = %d, host = %d\n", i, a[i], (-1 + i + 2*i));
	fail = 1;
      }
    }

    if (fail)
      printf ("Failed 33\n");
    else
      printf("Succeeded 33\n");
#endif

#if TEST34
    // Test: nowait
    fail = 0;
    for (int i = 0 ; i < N ; i++) {
      a[i] = -1;
      b[i] = i;
      c[i] = 2*i;
    }

    if (!cpuExec) {
#pragma omp target map(tofrom: a) map(to: b,c)
#pragma omp parallel
#pragma omp for simd nowait
    for (int i = 0 ; i < N ; i++)
      a[i] += b[i] + c[i];

    for (int i = 0 ; i < N ; i++)
      if (a[i] != (-1 + i + 2*i)) {
	printf("Error at %d: device = %d, host = %d\n", i, a[i], (-1 + i + 2*i));
	fail = 1;
      }
    }

    if (fail)
      printf ("Failed 34\n");
    else
      printf("Succeeded 34\n");
#endif

#if TEST35
    // Test: safelen
    fail = 0;
    for (int i = 0 ; i < N ; i++) {
      a[i] = -1;
      b[i] = i;
      c[i] = 2*i;
    }

    if (!cpuExec) {
#pragma omp target map(tofrom: a) map(to: b,c)
#pragma omp parallel
#pragma omp for simd safelen(16)
    for (int i = 0 ; i < N ; i++)
      a[i] += b[i] + c[i];

    for (int i = 0 ; i < N ; i++)
      if (a[i] != (-1 + i + 2*i)) {
	printf("Error at %d: device = %d, host = %d\n", i, a[i], (-1 + i + 2*i));
	fail = 1;
      }
    }

    if (fail)
      printf ("Failed 35\n");
    else
      printf("Succeeded 35\n");
#endif

#if TEST36
    // Test: simdlen
    fail = 0;
    for (int i = 0 ; i < N ; i++) {
      a[i] = -1;
      b[i] = i;
      c[i] = 2*i;
    }

    if (!cpuExec) {
#pragma omp target map(tofrom: a) map(to: b,c)
#pragma omp parallel
#pragma omp for simd simdlen(16)
    for (int i = 0 ; i < N ; i++)
      a[i] += b[i] + c[i];

    for (int i = 0 ; i < N ; i++)
      if (a[i] != (-1 + i + 2*i)) {
	printf("Error at %d: device = %d, host = %d\n", i, a[i], (-1 + i + 2*i));
	fail = 1;
      }
    }

    if (fail)
      printf ("Failed 36\n");
    else
      printf("Succeeded 36\n");
#endif

#if TEST37
    // Test: aligned
    fail = 0;
    for (int i = 0 ; i < N ; i++) {
      a[i] = -1;
      b[i] = i;
      c[i] = 2*i;
    }

    if (!cpuExec) {
#pragma omp target map(tofrom: a) map(to: b,c)
#pragma omp parallel
#pragma omp for simd aligned(a,b,c:8)
    for (int i = 0 ; i < N ; i++)
      a[i] += b[i] + c[i];

    for (int i = 0 ; i < N ; i++)
      if (a[i] != (-1 + i + 2*i)) {
	printf("Error at %d: device = %d, host = %d\n", i, a[i], (-1 + i + 2*i));
	fail = 1;
      }
    }

    if (fail)
      printf ("Failed 37\n");
    else
      printf("Succeeded 37\n");
#endif
  
  return 0;
}

#include <stdio.h>

#include "../utilities/check.h"

#define N 100

#pragma omp declare target
#pragma omp declare simd
int foo(int k) {
  return k+1;
}

#pragma omp declare simd simdlen(16)
int foo_simdlen(int k) {
  return k+1;
}

#pragma omp declare simd linear(b:1)
int foo_linear(int *b) {
  return *b+1;
}

#pragma omp declare simd aligned(b:8)
int foo_aligned(int *b) {
  return *b+1;
}

#pragma omp declare simd linear(b:1) uniform(c)
int foo_uniform(int *b, int c) {
  return *b+c;
}

#pragma omp declare simd linear(b:1) uniform(c) inbranch
int foo_inbranch(int *b, int c) {
  return *b+c;
}

#pragma omp declare simd linear(b:1) uniform(c) notinbranch
int foo_notinbranch(int *b, int c) {
  return *b+c;
}
#pragma omp end declare target



int main()
{
  check_offloading();

  int a[N], aa[N], b[N];
  int i, fail = 0;

  /// Test: no clauses

  // initialize
  for(i=0; i<N; i++)
    aa[i] = a[i] = -1;

  // offload
#pragma omp target map(tofrom: a[0:100])
  {
    int k;
    #pragma omp simd
    for(k=0; k<N; k++)
      a[k] = foo(k);
  }

  // host
  for(i=0; i<N; i++)
    aa[i] = i+1;

  // check
  for(i=0; i<N; i++) {
    if (a[i] != aa[i]) {
      printf("%d: a %d != %d\n", i, a[i], aa[i]);
      fail = 1;
    }
  }

  // report
  if (fail)
    printf("failed\n");
  else
    printf("success\n");

  /// Test: simdlen
  fail = 0;

  // initialize
  for(i=0; i<N; i++)
    aa[i] = a[i] = -1;

  // offload
  #pragma omp target map(tofrom: a[0:100])
  {
    int k;
    #pragma omp simd
    for(k=0; k<N; k++)
      a[k] = foo_simdlen(k);
  }

  // host
  for(i=0; i<N; i++)
    aa[i] = i+1;

  // check
  for(i=0; i<N; i++) {
    if (a[i] != aa[i]) {
      printf("%d: a %d != %d\n", i, a[i], aa[i]);
      fail = 1;
    }
  }

  // report
  if (fail)
    printf("failed\n");
  else
    printf("success\n");

  /// Test: linear
  fail = 0;

  // initialize
  for(i=0; i<N; i++) {
    aa[i] = a[i] = -1;
    b[i] = i;
  }

  // offload
  #pragma omp target map(tofrom: a[0:100]) map(to:b[:100])
  {
    int k;
    #pragma omp simd
    for(k=0; k<N; k++) {
      a[k] += foo_linear(&b[k]); // -1 += i
    }
  }

  // host
  for(i=0; i<N; i++)
    aa[i] = i;

  // check
  for(i=0; i<N; i++) {
    if (a[i] != aa[i]) {
      printf("%d: a %d != %d\n", i, a[i], aa[i]);
      fail = 1;
    }
  }

  // report
  if (fail)
    printf("failed\n");
  else
    printf("success\n");

  /// Test: aligned
  fail = 0;

  // initialize
  for(i=0; i<N; i++) {
    aa[i] = a[i] = -1;
    b[i] = i;
  }

  // offload
  #pragma omp target map(tofrom: a[0:100]) map(to:b[:100])
  {
    int k;
    #pragma omp simd
    for(k=0; k<N; k++) {
      a[k] += foo_aligned(&b[k]); // -1 += i
    }
  }

  // host
  for(i=0; i<N; i++)
    aa[i] = i;

  // check
  for(i=0; i<N; i++) {
    if (a[i] != aa[i]) {
      printf("%d: a %d != %d\n", i, a[i], aa[i]);
      fail = 1;
    }
  }

  // report
  if (fail)
    printf("failed\n");
  else
    printf("success\n");

  /// Test: uniform
  fail = 0;

  // initialize
  for(i=0; i<N; i++) {
    aa[i] = a[i] = -1;
    b[i] = i;
  }

  // offload
  #pragma omp target map(tofrom: a[0:100]) map(to:b[:100])
  {
    int k;
    int c = 3;
    #pragma omp simd
    for(k=0; k<N; k++) {
      a[k] += foo_uniform(&b[k],c); // -1 += i
    }
  }

  // host
  for(i=0; i<N; i++)
    aa[i] = i + 2;

  // check
  for(i=0; i<N; i++) {
    if (a[i] != aa[i]) {
      printf("%d: a %d != %d\n", i, a[i], aa[i]);
      fail = 1;
    }
  }

  // report
  if (fail)
    printf("failed\n");
  else
    printf("success\n");

  /// Test: inbranch
  fail = 0;

  // initialize
  for(i=0; i<N; i++) {
    aa[i] = a[i] = -1;
    b[i] = i;
  }

  // offload
  #pragma omp target map(tofrom: a[0:100]) map(to:b[:100])
  {
    int k;
    int c = 3;
    #pragma omp simd
    for(k=0; k<N; k++) {
      if (k%2 == 0)
	a[k] += foo_inbranch(&b[k],c); // -1 += i
    }
  }

  // host
  for(i=0; i<N; i++)
    if (i%2 == 0)
      aa[i] = i + 2;

  // check
  for(i=0; i<N; i++) {
    if (a[i] != aa[i]) {
      printf("%d: a %d != %d\n", i, a[i], aa[i]);
      fail = 1;
    }
  }

  // report
  if (fail)
    printf("failed\n");
  else
    printf("success\n");

  /// Test: notinbranch
  fail = 0;

  // initialize
  for(i=0; i<N; i++) {
    aa[i] = a[i] = -1;
    b[i] = i;
  }

  // offload
  #pragma omp target map(tofrom: a[0:100]) map(to:b[:100])
  {
    int k;
    int c = 3;
    #pragma omp simd
    for(k=0; k<N; k++) {
	a[k] += foo_notinbranch(&b[k],c); // -1 += i
    }
  }

  // host
  for(i=0; i<N; i++)
      aa[i] = i + 2;

  // check
  for(i=0; i<N; i++) {
    if (a[i] != aa[i]) {
      printf("%d: a %d != %d\n", i, a[i], aa[i]);
      fail = 1;
    }
  }

  // report
  if (fail)
    printf("failed\n");
  else
    printf("success\n");

  return 0;
}

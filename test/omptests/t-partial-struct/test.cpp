#include <stdio.h>
#include <omp.h>
#include <stdint.h>
#include <cstddef>

#include "../utilities/check.h"
#include "../utilities/utilities.h"

/*
export "LD_LIBRARY_PATH=/gsa/yktgsa/home/e/i/eichen/new-tlomp/lomp/source/lib64:/usr/local/cuda/lib64"
export LIBRARY_PATH="/home/eichen/eichen/lnew/obj/lib"

/gsa/yktgsa/home/e/i/eichen/lnew/obj/bin/clang++ -v  -I/gsa/yktgsa/home/e/i/eichen/new-tlomp/lomp/source/lib64/ -I/gsa/yktgsa/home/e/i/eichen/new-tlomp/lomp/source/   -L/gsa/yktgsa/home/e/i/eichen/new-tlomp/lomp/source/lib64/ -L/gsa/yktgsa/home/e/i/eichen/new-tlomp/lomp/source/lib64/ -target powerpc64le-ibm-linux-gnu -mcpu=pwr8 -fopenmp=libomp -fopenmp-targets=nvptx64-nvidia-cuda -O3 test.cpp
 */

#define N 99

#define CHECK             1
#define TEST_SIMPLE       1
#define TEST_SINGLE_ODD   1
#define TEST_SINGLE_EVEN  1
#define TEST_SINGLE       1
#define TEST_DOUBLE       1
#define TEST_SHADOW_PTR   1
#define TEST_NESTED       1

// from another test (nested)
typedef struct {
  int *a;
} SSS;

typedef struct {
  SSS *s;
} TTT;

// for the first batch of test
class S {
 public:
  int *p;
  int a;
  int b[N];
  int c;
  int d;
  int *q;
  int e[N];
  int f;

  S() {
    p = NULL;
    q = NULL;
  }

  void Init(int aa, int bb, int cc, int dd, int ee, int ff, int pp, int qq)
  {
    // malloc
    if(p && q) {
      free(p);
      free(q);
    }
    p = (int *)malloc(N*sizeof(int));
    q = (int *)malloc(N*sizeof(int));
    // init
    int i;
    for(i=0; i<N; i++) {
      b[i] = bb+ 2*i;
      e[i] = ee + 4*i;
      p[i] = pp+i;
      q[i] = qq + 3*i;
    }
    a = aa;
    c = cc;
    d = dd;
    f = ff;
  }

  int Verify(int aa, int bb, int cc, int dd, int ee, int ff, int pp, int qq)
  {
    int i, error = 0;
    if (a != aa) printf("a got %d, expected %d, error %d\n", a , aa, ++error);
    for(i=0; i<N; i++) {
      if (b[i] != bb + 2*i) printf("%d: b got %d, expected %d, error %d\n", i, b[i], bb + 2*i, ++error);
    }
    if (c != cc) printf("c got %d, expected %d, error %d\n", c , cc, ++error);
    if (d != dd) printf("d got %d, expected %d, error %d\n", d , dd, ++error);
    for(i=0; i<N; i++) {
      if (e[i] != ee + 4*i) printf("%d: e got %d, expected %d, error %d\n", i, e[i], ee + 4*i, ++error);
    }
    if (f != ff) printf("f got %d, expected %d, error %d\n", f , ff, ++error);
    for(i=0; i<N; i++) {
      if (p[i] != pp + 1*i) printf("%d: p got %d, expected %d, error %d\n", i, p[i], pp + 1*i, ++error);
    }
    for(i=0; i<N; i++) {
      if (q[i] != qq + 3*i) printf("%d: q got %d, expected %d, error %d\n", i, q[i], qq + 3*i, ++error);
    }
    return error;
  }
};

int main() {
  int8_t j = 0;
  int16_t i = 0;
  S s1, s2; 
  int totError = 0;
  int error = 0;

  #if CHECK
    check_offloading();
  #endif


//printf("offset of data struct S: p %lu (%lu mod 8), a %lu, b %lu, c %lu, d %lu, q %lu (%lu mod 8), e %lu, f %lu\n",
//       offsetof(S, p), offsetof(S, p) % 8, offsetof(S, a), offsetof(S, b), offsetof(S, c), offsetof(S, d),
//       offsetof(S, q), offsetof(S, q) % 8, offsetof(S, e), offsetof(S, f));    
  #if TEST_SIMPLE
    totError += error;
    s1.Init(1, 2, 3, 4, 5, 6, 7, 8);
    #pragma omp target map(tofrom: s1.b[0:N], s1.c, s1.d) 
    {
      int i;
      s1.c++;
      s1.d++;
      for(i=0; i<N; i++) s1.b[i]++;
    }
    error = s1.Verify(1, 2+1, 3+1, 4+1, 5, 6, 7, 8);
    printf("%s SIMPLE test with %d error(s)\n", (error ? "FAILED" : "Succeeded"), error); 
  #endif

  #if TEST_SINGLE_EVEN
    totError += error;
    s1.Init(1, 2, 3, 4, 5, 6, 7, 8);
    #pragma omp target map(tofrom: s1.c, s1.q[0:N]) 
    {
      int i;
      s1.c++;
      for(i=0; i<N; i++) s1.q[i]++;
      //printf("addr q from device: 0x%llx (%llu mod 8)\n", (unsigned long long) &s1.q, (unsigned long long) &s1.q % 8);
      //printf("addr q obj from device: 0x%llx\n", (unsigned long long) s1.q);
    }
    error = s1.Verify(1, 2, 3+1, 4, 5, 6, 7, 8+1);
    printf("%s SINGLE_EVEN test with %d error(s)\n", (error ? "FAILED" : "Succeeded"), error); 
  #endif

  #if TEST_SINGLE_ODD
    totError += error;
    s1.Init(1, 2, 3, 4, 5, 6, 7, 8);
    #pragma omp target map(tofrom: s1.d, s1.q[0:N]) 
    {
      int i;
      s1.d++;
      for(i=0; i<N; i++) s1.q[i]++;
      //printf("addr q from device: 0x%llx (%llu mod 8)\n", (unsigned long long) &s1.q, (unsigned long long) &s1.q % 8);
      //printf("addr q obj from device: 0x%llx\n", (unsigned long long) s1.q);
    }
    error = s1.Verify(1, 2, 3, 4+1, 5, 6, 7, 8+1);
    printf("%s SINGLE_ODD test with %d error(s)\n", (error ? "FAILED" : "Succeeded"), error); 
  #endif

  #if TEST_SINGLE
    totError += error;
    s1.Init(1, 2, 3, 4, 5, 6, 7, 8);
    #pragma omp target map(tofrom: s1.c, s1.d, s1.q[0:N]) 
    {
      int i;
      s1.c++;
      s1.d++;
      for(i=0; i<N; i++) s1.q[i]++;
      //printf("addr q from device: 0x%llx (%llu mod 8)\n", (unsigned long long) &s1.q, (unsigned long long) &s1.q % 8);
      //printf("addr q obj from device: 0x%llx\n", (unsigned long long) s1.q);
    }
    error = s1.Verify(1, 2, 3+1, 4+1, 5, 6, 7, 8+1);
    printf("%s SINGLE test with %d error(s)\n", (error ? "FAILED" : "Succeeded"), error); 
  #endif


  #if TEST_DOUBLE
    totError += error;
    s1.Init(1, 2, 3, 4, 5, 6, 7, 8);
    s2.Init(11, 12, 13, 14, 15, 16, 17, 18);
    #pragma omp target map(tofrom: s1.c, s1.d, s1.p[0:N], s2.b, s2.d, s2.q[0:N]) 
    {
      //printf("s1: addr p from device: 0x%llx (%llu mod 8)\n", (unsigned long long) &s1.p, (unsigned long long) &s1.p % 8);
      //printf("s1: addr p obj from device: 0x%llx\n", (unsigned long long) s1.p);
      //printf("s2: addr q from device: 0x%llx (%llu mod 8)\n", (unsigned long long) &s2.q, (unsigned long long) &s2.q % 8);
      //printf("s2: addr q obj from device: 0x%llx\n", (unsigned long long) s2.q);

      int i;
      // s1
      s1.c++;
      s1.d++;
      for(i=0; i<N; i++) s1.p[i]++;
      // s2
      for(i=0; i<N; i++) s2.b[i]++;
      s2.d++;
      for(i=0; i<N; i++) s2.q[i]++;
    }
    error  = s1.Verify(1, 2, 3+1, 4+1, 5, 6, 7+1, 8);
    error += s2.Verify(11, 12+1, 13, 14+1, 15, 16, 17, 18+1);
    printf("%s DOUBLE test with %d error(s)\n", (error ? "FAILED" : "Succeeded"), error); 
  #endif

  #if TEST_SHADOW_PTR
    totError += error;
    s1.Init(1, 2, 3, 4, 5, 6, 7, 8);
    #pragma omp target data map(tofrom: s1)
    #pragma omp target data map(tofrom: s1.q[0:N])
    {
      s1.c++;
      #pragma omp target update to(s1)

      #pragma omp target map(tofrom: s1.c, s1.q[0:N])
      {
        int i;
        for(i=0; i<N; i++) s1.q[i]++;
      }

      #pragma omp target update from(s1)
    }
    error = s1.Verify(1, 2, 3+1, 4, 5, 6, 7, 8+1);
    printf("%s SHADOW_PTR test with %d error(s)\n", (error ? "FAILED" : "Succeeded"), error); 
  #endif

  #if TEST_NESTED
    totError += error;

    TTT t;
    t.s = new SSS();
    t.s->a = (int *) malloc(N*sizeof(int));
  
    //printf("addr of t 0x%llx, t.s.a 0x%llx, t.s->a[]: 0x%llx\n", 
    //  (long long)&t, (long long) &t.s->a, (long long) &t.s->a[0]);
    //#pragma omp target data map(from: t)
    #pragma omp target map(from: t.s->a[:N])
    #pragma omp teams distribute parallel for
    for(int i = 0 ; i < N ; i++) {
      t.s->a[i] = i;
    }

    error = 0;
    for(int i = 0 ; i < N ; i++) {
      if(t.s->a[i] != i) printf("Error at %d, %d, error %d\n", i, t.s->a[i], ++error);
    }
    printf("%s NESTED test with %d error(s)\n", (error ? "FAILED" : "Succeeded"), error); 
  #endif

  totError += error;
  if (totError==0) printf("success\n"); else printf("completed with some errors\n");
  return 1;
}

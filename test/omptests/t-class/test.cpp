#define TEST_A0 1
#define TEST_A1 1
#define TEST_A2 0 /* currently a bug */
#define TEST_B  1
#define TEST_C  0  /* currently a missing feature */


/*
export "LD_LIBRARY_PATH=/gsa/yktgsa/home/e/i/eichen/new-tlomp/lomp/source/lib64:/usr/local/cuda/lib64"
export LIBRARY_PATH="/home/eichen/eichen/lnew/obj/lib"

/gsa/yktgsa/home/e/i/eichen/lnew/obj/bin/clang++ -v  -I/gsa/yktgsa/home/e/i/eichen/new-tlomp/lomp/source/lib64/ -I/gsa/yktgsa/home/e/i/eichen/new-tlomp/lomp/source/   -L/gsa/yktgsa/home/e/i/eichen/new-tlomp/lomp/source/lib64/ -L/gsa/yktgsa/home/e/i/eichen/new-tlomp/lomp/source/lib64/ -target powerpc64le-ibm-linux-gnu -mcpu=pwr8 -fopenmp=libomp -fopenmp-targets=nvptx64-nvidia-cuda -O3 test.cpp
 */
#include <stdlib.h>
#include <stdio.h>


#define N 1000

#if TEST_A0
// A one target into a regular class, using class data
class A0 {
public:
  int a, b, c, sum, *p;
  A0() {
    a = 1; b = 2; c = 3;
    p = (int *)malloc(N*sizeof(int));
    for(int i=0; i<N; i++) p[i] = 10+i;
  }
  ~A0() {
    free(p);
  }

  int Num() {
    #pragma omp target 
    for(int i=0; i<N; i++)
    {
      sum = a + b + c;
    }
    return sum;
  }
};
#endif

#if TEST_A1
// A1 one target into a regular class, using class data
class A1 {
public:
  int a, b, c, sum, *p;
  A1() {
    a = 1; b = 2; c = 3;
    p = (int *)malloc(N*sizeof(int));
    for(int i=0; i<N; i++) p[i] = 10+i;
  }
  ~A1() {
    free(p);
  }

  int Num() {
  #pragma omp target teams distribute parallel for map(a, b, c, p[:N]) 
    for(int i=0; i<N; i++)
    {
      p[i] = a + b + c;
      //printf("%d : %d\n", i, p[i]);
    }
    return p[0];
  }
};
#endif

#if TEST_A2
// A1 one target into a regular class, uing class data
class A2 {
public:
  int a, b, c, sum, *p;
  A2() {
    a = 1; b = 2; c = 3;
    p = (int *)malloc(N*sizeof(int));
    for(int i=0; i<N; i++) p[i] = 10+i;
  }
  ~A2() {
    free(p);
  }

  int Num() {
  #pragma omp target
  #pragma omp parallel
    {
      sum = a + b + c;
      //printf("%d : %d\n", i, p[i]);
    }
    return sum;
  }
};
#endif

#if TEST_B
// B entire class on target
#pragma omp declare target   
class B {
public:
  int a, b, c;
  B() {
    a = 1; b = 2; c = 3;
  }
};

B bb;

int foo() {
  return bb.a + bb.b + bb.c;
}
#pragma omp end declare target   
#endif

#if TEST_C
class C {
public:
  int a, b, c, sum, *p;

  C() {
    a = 1; b = 2; c = 3;
    p = (int *)malloc(4*sizeof(int));
    for(int i=0; i<4; i++) p[i] = 10+i;
  }
  ~C() {
    free(p);
  }

  #pragma omp declare target   
  int Num() {
    {
      sum = a + b + c;
    }
    return sum;
  }
  #pragma omp end declare target   
};
#endif

int main() {

#if TEST_A0
  A0 a0;
  printf("test a0: sum is %d\n", a0.Num());
#endif

#if TEST_A1
  A1 a1;
  printf("test a1: sum is %d\n", a1.Num());
#endif

#if TEST_A2
  A2 a2;
  printf("test a2: sum is %d\n", a2.Num());
#endif

  #pragma omp target
  printf("test b: foo is %d\n", foo());

#if TEST_C
  #pragma omp target
  {
    C c;
    c.a = 1; c.b = 2; c.c = 3;
    printf("test c: c.foo is %d\n", c.foo());
  }
#endif

  return 1;
}

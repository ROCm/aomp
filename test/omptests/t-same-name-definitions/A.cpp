#include "AB.h"

void Hello(const char *N, int V) {
  printf("Hello from %s(v%d)\n", N, V);
}

void fAB1() {
 #pragma omp target
 Hello("fAB1");
}

#pragma omp declare target

int AB1 = 1;
ABTy<3> AB3;

int AA1 = 1;
static int AA2 = 2;
ABTy<100> AA3;
static ABTy<100> AA4;

static int AB2 = 2; 
static ABTy<4> AB4;

#pragma omp end declare target

static void fAB2() {
 #pragma omp target
 Hello("fAB2");
}

template<int X>
static void fAB4() {
 #pragma omp target
 Hello("fAB4", X);
}

void fAA1() {
 #pragma omp target
 Hello("fAA1");
}

static void fAA2() {
 #pragma omp target
 Hello("fAA2");
}

template<int X>
void fAA3() {
 #pragma omp target
 Hello("fAA3", X);
}

template<int X>
static void fAA4() {
 #pragma omp target
 Hello("fAA4", X);
}

void a() {
  fAB1();
  fAB2();
  fAB3<100>();
  fAB4<200>();
  
  fAA1();
  fAA2();
  fAA3<100>();
  fAA4<200>();
  
  #pragma omp target
  {
    printf("A --> AB1 %d\n", AB1++);
    printf("A --> AB2 %d\n", AB2++);
    printf("A --> AB3.Val %d\n", AB3.Val++);
    printf("A --> AB4.Val %d\n", AB4.Val++);
    printf("A --> AA1 %d\n", AA1++);
    printf("A --> AA2 %d\n", AA2++);
    printf("A --> AA3.Val %d\n", AA3.Val++);
    printf("A --> AA4.Val %d\n", AA4.Val++);
  }
}

int main() {
  a();
  b();
  return 0;
}


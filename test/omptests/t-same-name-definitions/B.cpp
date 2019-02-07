#include "AB.h"

#pragma omp declare target

extern int AB1;
extern ABTy<3> AB3;


static int AA1 = 1;
static int AA2 = 2;
static ABTy<100> AA3;
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

static void fAA1() {
 #pragma omp target
 Hello("fAA1b");
}

static void fAA2() {
 #pragma omp target
 Hello("fAA2b");
}

template<int X>
static void fAA3() {
 #pragma omp target
 Hello("fAA3b", X);
}

template<int X>
static void fAA4() {
 #pragma omp target
 Hello("fAA4b", X);
}

void b() {
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
    printf("B --> AB1 %d\n", AB1++);
    printf("B --> AB2 %d\n", AB2++);
    printf("B --> AB3.Val %d\n", AB3.Val++);
    printf("B --> AB4.Val %d\n", AB4.Val++);
    printf("B --> AA1 %d\n", AA1++);
    printf("B --> AA2 %d\n", AA2++);
    printf("B --> AA3.Val %d\n", AA3.Val++);
    printf("B --> AA4.Val %d\n", AA4.Val++);
  }

}
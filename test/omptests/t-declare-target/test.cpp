#include <stdio.h>
#include <omp.h>
#include "../utilities/check.h"

//#######################################
//#######################################
//#######################################

#pragma omp declare target

int t1_d1(int a) {
  return a+1;
}

static int t1_d2(int a) {
  return a+1;
}

template<typename T>
T t1_d3(T a) {
  return a+1;
}

template<typename T>
static T t1_d4(T a) {
  return a+1;
}

int t1_G1 = 2;

#pragma omp end declare target

int t1_G2 = 3;

void t1(int a /* = 1 */) {
  int A[1] = {0};

  #pragma omp target
  {
    A[0] = a + t1_d1(A[0]) + t1_d2(A[0]) + t1_d3(A[0]) + t1_d4(A[0]) + t1_G1 + t1_G2;
  }
  
  int Expected = 1 + 1 + 1 + 1 + 1 + 2 + 3;
  
  if (A[0] != Expected)
    printf("Error %d != %d\n",A[0], Expected);
  else
    printf("Success!\n");
}

//#######################################
//#######################################
//#######################################

#pragma omp declare target

struct t2_d1 {
  int Val;
  
  void inc() {
    ++Val;
  }
};

template<typename T>
struct t2_d2 {
  T Val;
  
  void inc() {
    ++Val;
  }
};

#pragma omp end declare target

void t2(int a /* = 1 */) {
  t2_d1 A;
  t2_d2<int> B;
  
  A.Val = 2;
  B.Val = 3;

  #pragma omp target
  { 
    A.inc();
    B.inc();
    A.Val += B.Val; 
  }
  
  int Expected = 2 + 1 + 3 + 1;
  
  if (A.Val != Expected)
    printf("Error %d != %d\n",A.Val, Expected);
  else
    printf("Success!\n");
}

//#######################################
//#######################################
//#######################################

int main(int argc, char *argv[]) {
  check_offloading();
  
  t1(argc ? 1 : 5);
  t2(argc ? 1 : 5);
  
  printf("Done!\n");
  return 0;
}


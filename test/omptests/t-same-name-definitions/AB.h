#include "stdio.h"
#include "omp.h"

#pragma omp declare target to(omp_is_initial_device)

#pragma omp declare target

void Hello(const char *N, int V = 0);

template<int val>
struct ABTy {
  int Dummy = 0;
  int Val = val;
};

// static int AB2 = 2; 
// static ABTy<4> AB4;

#pragma omp end declare target

void fAB1();

// static void fAB2() {
//  #pragma omp target
//  Hello("fAB2");
// }

template<int X>
void fAB3() {
 #pragma omp target
 Hello("fAB3", X);
}
// 
// template<int X>
// static void fAB4() {
//  #pragma omp target
//  Hello("fAB4", X);
// }


void a();
void b();

#include <iostream>
#include <stdio.h>


// template <typename LOOP_BODY>
// inline
// void forall(int begin, int end, LOOP_BODY loop_body)
// {
//   #pragma omp target
//   {
//   #pragma omp parallel for schedule(static)
//      for ( int ii = begin ; ii < end ; ++ii ) 
//      {
//         loop_body( ii );
//      }
//   }
// }

#define N (1000)

#pragma omp declare target
struct S1 {
  int A;
  int B;
  virtual void foo() {
    ++A;
    ++B;
  }
};

struct S2 : public S1 {
  void foo() override {
    --A;
    --B;
  }
  void bar () {
    A -= 2;
    B -= 2;
  }
};
#pragma omp end declare target

//
// Demonstration of the RAJA abstraction using lambdas
// Requires data mapping onto the target section
//
int main() {
  S2 s;
  s.A = 10;
  s.B = 100;
  
  s.foo();
  
  #pragma omp target map(s)
  {
    s.bar();
  } 
  
  printf("%d %d\n", s.A, s.B); 
  return 0;
}
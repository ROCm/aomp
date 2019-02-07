
#include <iostream>

template <typename LOOP_BODY>
inline
void forall(int begin, int end, LOOP_BODY loop_body)
{
#pragma omp target
#pragma omp parallel for schedule(static)
   for ( int ii = begin ; ii < end ; ++ii ) {
      loop_body( ii );
   }
}


#define N (1000)

//
// Demonstration of the RAJA abstraction using lambdas
// Requires data mapping onto the target section
//
int main() {
  double a[N], b[N];
  double c[N], d[N];

  for (int i = 0; i < N; i++) {
    a[i] = i+1;
    b[i] = -i;
    c[i] = -9;
  }

  #pragma omp target data map(tofrom: c[0:N]) map(to: a[0:N], b[0:N])
  {

  forall (0, N,
   [&] (int i) {
    c[i] += a[i]+b[i];
   }
  );

  }

  int fail = 0;
  for (int i = 0; i < N; i++) {
    if (c[i] != -8) {
      std::cout << "Failed at " << i << " with val " << c[i] << std::endl;
      fail = 1;
    }
  }

  if (fail) {
    std::cout << "Failed" << std::endl;
  } else {
    std::cout << "Succeeded" << std::endl;
  }

  return c[2] + c[5];

}


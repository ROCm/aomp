#include<omp.h>

// This test checks if the debug infomation were correctly
// mapped for cross-tream reduction code.

int main()
{
  int N = 100000;
  double x[N];
  double sum = 0.0;

  #pragma omp target teams distribute parallel for reduction(+: sum)
  for (int i=0; i<N; i++){
    sum += x[i];
  }

  sum = sum/(double)N;
}

/// CHECK:      firstprivate(N)[4] (implicit)
/// CHECK-NEXT: tofrom(sum)[8] (implicit)
/// CHECK-NEXT: firstprivate(unknown)[8] (implicit)
/// CHECK-NEXT: tofrom(x)[800000] (implicit)
/// CHECK-NEXT: firstprivate(unknown)[8]
/// CHECK-NEXT: firstprivate(unknown)[8]

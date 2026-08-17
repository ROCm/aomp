#include<omp.h>

// This test checks if the debug information is correctly mapped for
// cross-team reduction code.
//
// The kernel takes exactly the four implicit arguments below. The downstream
// Xteam reduction implementation used to append two more (the per-team value
// buffer and the teams-done counter); it has been removed, so those two extra
// 'firstprivate(unknown)[8]' entries must no longer show up.

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

/// CHECK:      Entering OpenMP kernel at xteam_red_debug_info.c:17:3 with 4 arguments:
/// CHECK-NEXT: firstprivate(N)[4] (implicit)
/// CHECK-NEXT: tofrom(sum)[8] (implicit)
/// CHECK-NEXT: firstprivate(unknown)[8] (implicit)
/// CHECK-NEXT: tofrom(x)[800000] (implicit)
/// CHECK-NOT:  firstprivate(unknown)

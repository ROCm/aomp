
#include <stdio.h>
#include <omp.h>

#include "../utilities/check.h"
#include "../utilities/utilities.h"

#define TRIALS (1)

#define N (1024)

#define INIT() INIT_LOOP(N, {C[i] = 1; D[i] = i; E[i] = -i;})

#define ZERO(X) ZERO_ARRAY(N, X) 

#define VAR1(i) double A##i[1]; A##i[0] = 1;
#define VAR10(x) VAR1(x##0) VAR1(x##1) VAR1(x##2) VAR1(x##3) VAR1(x##4) VAR1(x##5) VAR1(x##6) VAR1(x##7) VAR1(x##8) VAR1(x##9)
#define VAR100(x)  VAR10(x##0) VAR10(x##1) VAR10(x##2) VAR10(x##3) VAR10(x##4) VAR10(x##5) VAR10(x##6) VAR10(x##7) VAR10(x##8) VAR10(x##9)
#define VAR1000  VAR100(0) VAR100(1) VAR100(2) VAR100(3) VAR100(4) VAR100(5) VAR100(6) VAR100(7) VAR100(8) VAR100(9)

#define SUM10(x) A##x##0[0] + A##x##1[0] + A##x##2[0] + A##x##3[0] + A##x##4[0] + A##x##5[0] + A##x##6[0] + A##x##7[0] + A##x##8[0] + A##x##9[0]
#define SUM50(x) SUM10(x##0) + SUM10(x##1) + SUM10(x##2) + SUM10(x##3) + SUM10(x##4)
#define SUM100(x) SUM10(x##0) + SUM10(x##1) + SUM10(x##2) + SUM10(x##3) + SUM10(x##4) + SUM10(x##5) + SUM10(x##6) + SUM10(x##7) + SUM10(x##8) + SUM10(x##9)
#define SUM200 SUM100(0) + SUM100(1)
#define SUM400 SUM200 + SUM100(2) + SUM100(3)

int main(void) {
  check_offloading();

  double A[N], B[N], C[N], D[N], E[N];

  INIT();

  VAR1000;

  //
  // Test: Multiple parallel regions with varying and large
  // numbers of captured arguments.
  //
  TEST({
   for (int i = 0; i < 1024; i++) {
     A[i] = 0;
   }
   _Pragma("omp parallel for")
   for (int i = 0; i < 1024; i++) {
     A[i] += SUM10(00);
   }
   _Pragma("omp parallel for")
   for (int i = 0; i < 1024; i++) {
     A[i] += SUM50(0);
   }
//   _Pragma("omp parallel for")
//   for (int i = 0; i < 1024; i++) {
//     A[i] += SUM200;
//   }
//   _Pragma("omp parallel for")
//   for (int i = 0; i < 1024; i++) {
//     A[i] += SUM400;
//   }
   _Pragma("omp parallel for")
   for (int i = 0; i < 1024; i++) {
     A[i] += SUM10(00);
   }
  }, VERIFY(0, 1024, A[i], 70 ));

  return 0;
}

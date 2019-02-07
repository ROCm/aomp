
#include <stdio.h>
#include <float.h>
#include <stdlib.h>
#include <math.h>
#include <omp.h>

#include "../utilities/check.h"
#include "../utilities/utilities.h"

#define TRIALS (1)

#define N (957*3)

#define ZERO(X) ZERO_ARRAY(N, X) 

#define INIT() { \
INIT_LOOP(N, { \
  Ac[i] = i % 100 == 0 ? 1 : 0; \
  Bc[i] = i << 4; \
  Cc[i] = -(i << 4); \
  Dc[i] = (2*i+1) << 4; \
  Ec[i] = (i % 2 == 0 ? 0x1 : 0x0) | \
          (i % 3 == 0 ? 0x2 : 0x0); \
  As[i] = 1; \
  Bs[i] = i << 8; \
  Cs[i] = -(i << 8); \
  Ds[i] = (2*i+1) << 8; \
  Es[i] = ((i % 2 == 0 ? 0x1 : 0x0) << 4) | \
          ((i % 3 == 0 ? 0x2 : 0x0) << 4); \
  Ai[i] = 1 << 16; \
  Bi[i] = i << 16; \
  Ci[i] = -(i << 16); \
  Di[i] = (2*i+1) << 16; \
  Ei[i] = ((i % 2 == 0 ? 0x1 : 0x0) << 16) | \
          ((i % 3 == 0 ? 0x2 : 0x0) << 16); \
  All[i] = 1ll << 32; \
  Bll[i] = (long long) (i) << 32; \
  Cll[i] = -((long long) (i) << 32); \
  Dll[i] = ((long long) (2*i+1)) << 32; \
  Ell[i] = ((i % 2 == 0 ? 0x1ll : 0x0) << 32) | \
          ((i % 3 == 0 ? 0x2ll : 0x0) << 32); \
  Af[i] = 1 << 8; \
  Bf[i] = i << 8; \
  Cf[i] = -(i << 8); \
  Df[i] = (2*i+1) << 8; \
  Ef[i] = ((i % 2 == 0 ? 0x1 : 0x0) << 8) | \
          ((i % 3 == 0 ? 0x2 : 0x0) << 8); \
  Ad[i] = 1 << 16; \
  Bd[i] = i << 16; \
  Cd[i] = -(i << 16); \
  Dd[i] = (2*i+1) << 16; \
  Ed[i] = ((i % 2 == 0 ? 0x1 : 0x0) << 16) | \
          ((i % 3 == 0 ? 0x2 : 0x0) << 16); \
}) \
}

#define INIT1 (1)
#define INIT2 (3)
#define INIT3 (5)
#define INIT4 (7)
#define INITc5 (9)
#define INITs5 (9 << 4)
#define INITi5 (9 << 16)
#define INITll5 (9ll << 32)
#define INITf5 (9 << 8)
#define INITd5 (9 << 16)
#define INITc6 (0xf)
#define INITs6 (0xff << 4)
#define INITi6 (0xff << 16)
#define INITll6 (0xffll << 32)
#define INITf6 (0xff << 8)
#define INITd6 (0xff << 16)
#define INIT7 (0)
#define INIT8 (0)
#define INIT9 (1)
#define INIT10 (0)

#define EXPECTED_1 ( \
INIT1 + INIT2 + \
(1+N/100) + (1+N/100) + \
/* + (2*(N-1)+1) - (N-1) */ + \
(INITc5*2) + \
/* bitwise-and reduction should remove all bits from initial value except the LSB */ \
(0x1) + \
(0x3) + \
/* XOR: true if the number of variables with the value true is odd */ \
(0x2) + \
0 + 1 \
)

#define EXPECTED_2 ( \
INIT1 + INIT2 + \
N + N + \
/* + (2*(N-1)+1) - (N-1) */ + \
(INITs5*2*2*2) + \
/* bitwise-and reduction should remove all bits from initial value except the LSB */ \
(0x1 << 4) + \
(0x7 << 4) + \
/* XOR: true if the number of variables with the value true is odd */ \
(0x2 << 4) + \
1 + 1 \
)

#define EXPECTED_3 ( \
INIT1 + INIT2 + \
(N << 16) + (N << 16) + \
/* + (2*(N-1)+1) - (N-1) */ + \
(INITi5*2*2*2) + \
/* bitwise-and reduction should remove all bits from initial value except the LSB */ \
(0x1 << 16) + \
(0x7 << 16) + \
/* XOR: true if the number of variables with the value true is odd */ \
(0x2 << 16) + \
1 + 1 \
)

#define EXPECTED_4 ( \
INIT1 + INIT2 + \
((long long) N << 32) + ((long long) N << 32) + \
/* + (2*(N-1)+1) - (N-1) */ + \
(INITll5*2*2*2) + \
/* bitwise-and reduction should remove all bits from initial value except the LSB */ \
(0x1ll << 32) + \
(0x7ll << 32) + \
/* XOR: true if the number of variables with the value true is odd */ \
(0x2ll << 32) + \
1 + 1 \
)

#define EXPECTED_5 ( \
INIT1 + INIT2 + \
(N << 8) + (N << 8) + \
/* + (2*(N-1)+1) - (N-1) */ + \
(INITf5*2*2*2) + \
1 + 1 \
)

#define EXPECTED_6 ( \
INIT1 + INIT2 + \
(N << 16) + (N << 16) + \
/* + (2*(N-1)+1) - (N-1) */ + \
(INITd5*2*2*2) + \
1 + 1 \
)

#define REDUCTION_CLAUSES reduction(+:Rc1) reduction(-:Rc2) reduction(*:Rc5) \
                  reduction(&:Rc6) reduction(|:Rc7) reduction(^:Rc8) \
                  reduction(&&:Rc9) reduction(||:Rc10) \
                  reduction(+:Rs1) reduction(-:Rs2) reduction(*:Rs5) \
                  reduction(&:Rs6) reduction(|:Rs7) reduction(^:Rs8) \
                  reduction(&&:Rs9) reduction(||:Rs10) \
                  reduction(+:Ri1) reduction(-:Ri2) reduction(*:Ri5) \
                  reduction(&:Ri6) reduction(|:Ri7) reduction(^:Ri8) \
                  reduction(&&:Ri9) reduction(||:Ri10) \
                  reduction(+:Rll1) reduction(-:Rll2) reduction(*:Rll5) \
                  reduction(&:Rll6) reduction(|:Rll7) reduction(^:Rll8) \
                  reduction(&&:Rll9) reduction(||:Rll10) \
                  reduction(+:Rf1) reduction(-:Rf2) reduction(*:Rf5) \
                  reduction(&&:Rf9) reduction(||:Rf10) \
                  reduction(+:Rd1) reduction(-:Rd2) reduction(*:Rd5) \
                  reduction(&&:Rd9) reduction(||:Rd10)
//reduction(max:Ri3) reduction(min:Ri4)

#define REDUCTION_MAP map(tofrom: Rc1, Rc2, Rc5, Rc6, Rc7, Rc8, Rc9, Rc10) \
                      map(tofrom: Rs1, Rs2, Rs5, Rs6, Rs7, Rs8, Rs9, Rs10) \
                      map(tofrom: Ri1, Ri2, Ri5, Ri6, Ri7, Ri8, Ri9, Ri10) \
                      map(tofrom: Rll1, Rll2, Rll5, Rll6, Rll7, Rll8, Rll9, Rll10) \
                      map(tofrom: Rf1, Rf2, Rf5, Rf9, Rf10) \
                      map(tofrom: Rd1, Rd2, Rd5, Rd9, Rd10)

#define REDUCTION_INIT() {           \
      Rc1 = INIT1; Rc2 = INIT2;      \
      Rc3 = INIT3; Rc4 = INIT4;      \
      Rc5 = INITc5; Rc6 = INITc6;    \
      Rc7 = INIT7; Rc8 = INIT8;      \
      Rc9 = INIT9; Rc10 = INIT10;    \
                                     \
      Rs1 = INIT1; Rs2 = INIT2;      \
      Rs3 = INIT3; Rs4 = INIT4;      \
      Rs5 = INITs5; Rs6 = INITs6;    \
      Rs7 = INIT7; Rs8 = INIT8;      \
      Rs9 = INIT9; Rs10 = INIT10;    \
                                     \
      Ri1 = INIT1; Ri2 = INIT2;      \
      Ri3 = INIT3; Ri4 = INIT4;      \
      Ri5 = INITi5; Ri6 = INITi6;    \
      Ri7 = INIT7; Ri8 = INIT8;      \
      Ri9 = INIT9; Ri10 = INIT10;    \
                                     \
      Rll1 = INIT1; Rll2 = INIT2;    \
      Rll3 = INIT3; Rll4 = INIT4;    \
      Rll5 = INITll5; Rll6 = INITll6;\
      Rll7 = INIT7; Rll8 = INIT8;    \
      Rll9 = INIT9; Rll10 = INIT10;  \
                                     \
      Rf1 = INIT1; Rf2 = INIT2;      \
      Rf3 = INIT3; Rf4 = INIT4;      \
      Rf5 = INITf5; Rf6 = INITf6;    \
      Rf7 = INIT7; Rf8 = INIT8;      \
      Rf9 = INIT9; Rf10 = INIT10;    \
                                     \
      Rd1 = INIT1; Rd2 = INIT2;      \
      Rd3 = INIT3; Rd4 = INIT4;      \
      Rd5 = INITd5; Rd6 = INITd6;    \
      Rd7 = INIT7; Rd8 = INIT8;      \
      Rd9 = INIT9; Rd10 = INIT10;    \
}

#define REDUCTION_BODY() \
      Rc1 += Ac[i] + (Bc[i] + Cc[i]); \
      Rc2 += Ac[i] + (Bc[i] + Cc[i]); \
      /*Rc3 = Dc[i] > Rc3 ? Dc[i] : Rc3; \
      Rc4 = Cc[i] < Rc4 ? Cc[i] : Rc4; \*/ \
      Rc5 *= i == 2000 ? 2 : 1; \
      Rc6 &= ~(1 << (1 + i / 410)); \
      Rc7 |= 1 << (i / 2000); \
      Rc8 ^= Ec[i]; \
      Rc9 = Rc9 && Ac[i] > 0; \
      Rc10 = Rc10 || Ac[i] > 0; \
                                           \
      Rs1 += As[i] + (Bs[i] + Cs[i]); \
      Rs2 += As[i] + (Bs[i] + Cs[i]); \
      /*Rs3 = Ds[i] > Rs3 ? Ds[i] : Rs3; \
      Rs4 = Cs[i] < Rs4 ? Cs[i] : Rs4; \*/ \
      Rs5 *= i % 1000 == 0 ? 2 : 1; \
      Rs6 &= ~(1 << (5 + i / 410)); \
      Rs7 |= 1 << (4 + i / 1000); \
      Rs8 ^= Es[i]; \
      Rs9 = Rs9 && As[i] > 0; \
      Rs10 = Rs10 || As[i] > 0; \
                                          \
      Ri1 += Ai[i] + (Bi[i] + Ci[i]); \
      Ri2 += Ai[i] + (Bi[i] + Ci[i]); \
      /*Ri3 = Di[i] > Ri3 ? Di[i] : Ri3; \
      Ri4 = Ci[i] < Ri4 ? Ci[i] : Ri4; \*/ \
      Ri5 *= i % 1000 == 0 ? 2 : 1; \
      Ri6 &= ~(1 << (17 + i / 410)); \
      Ri7 |= 1 << (16 + i / 1000); \
      Ri8 ^= Ei[i]; \
      Ri9 = Ri9 && Ai[i] > 0; \
      Ri10 = Ri10 || Ai[i] > 0; \
                                          \
      Rll1 += All[i] + (Bll[i] + Cll[i]); \
      Rll2 += All[i] + (Bll[i] + Cll[i]); \
      /*Rll3 = Dll[i] > Rll3 ? Dll[i] : Rll3; \
      Rll4 = Cll[i] < Rll4 ? Cll[i] : Rll4; \*/ \
      Rll5 *= i % 1000 == 0 ? 2 : 1; \
      Rll6 &= ~(1ll << (33 + i / 410)); \
      Rll7 |= 1ll << (32 + i / 1000); \
      Rll8 ^= Ell[i]; \
      Rll9 = Rll9 && All[i] > 0; \
      Rll10 = Rll10 || All[i] > 0; \
                                         \
      Rf1 += Af[i] + (Bf[i] + Cf[i]); \
      Rf2 += Af[i] + (Bf[i] + Cf[i]); \
      /*Rf3 = Df[i] > Rf3 ? Df[i] : Rf3; \
      Rf4 = Cf[i] < Rf4 ? Cf[i] : Rf4; \*/ \
      Rf5 *= i % 1000 == 0 ? 2 : 1; \
      Rf9 = Rf9 && Af[i] > 0; \
      Rf10 = Rf10 || Af[i] > 0; \
                                        \
      Rd1 += Ad[i] + (Bd[i] + Cd[i]); \
      Rd2 += Ad[i] + (Bd[i] + Cd[i]); \
      /*Rd3 = Dd[i] > Rd3 ? Dd[i] : Rd3; \
      Rd4 = Cd[i] < Rd4 ? Cd[i] : Rd4; \*/ \
      Rd5 *= i % 1000 == 0 ? 2 : 1; \
      Rd9 = Rd9 && Ad[i] > 0; \
      Rd10 = Rd10 || Ad[i] > 0;

#define REDUCTION_LOOP() \
    for (int i = 0; i < N; i++) { \
      REDUCTION_BODY(); \
    }

#define REDUCTION_FINAL() { \
      OUT[0] += Rc1 + Rc2 /*+ Rc3 + Rc4 */ + Rc5 + Rc6 + Rc7 + Rc8 + Rc9 + Rc10; \
      OUT[1] += Rs1 + Rs2 /*+ Rs3 + Rs4 */ + Rs5 + Rs6 + Rs7 + Rs8 + Rs9 + Rs10; \
      OUT[2] += Ri1 + Ri2 /*+ Ri3 + Ri4 */ + Ri5 + Ri6 + Ri7 + Ri8 + Ri9 + Ri10; \
      OUT[3] += Rll1 + Rll2 /*+ Rll3 + Rll4 */ + Rll5 + Rll6 + Rll7 + Rll8 + Rll9 + Rll10; \
      OUT[4] += (long long) (Rf1 + Rf2 /*+ Rf3 + Rf4 */ + Rf5 + Rf9 + Rf10); \
      OUT[5] += (long long) (Rd1 + Rd2 /*+ Rd3 + Rd4 */ + Rd5 + Rd9 + Rd10); \
}

int main(void) {
  check_offloading();

  char Ac[N], Bc[N], Cc[N], Dc[N], Ec[N];
  short As[N], Bs[N], Cs[N], Ds[N], Es[N];
  int Ai[N], Bi[N], Ci[N], Di[N], Ei[N];
  long long All[N], Bll[N], Cll[N], Dll[N], Ell[N];
  float Af[N], Bf[N], Cf[N], Df[N], Ef[N];
  double Ad[N], Bd[N], Cd[N], Dd[N], Ed[N];
  char Rc1, Rc2, Rc3, Rc4, Rc5, Rc6, Rc7, Rc8, Rc9, Rc10;
  short Rs1, Rs2, Rs3, Rs4, Rs5, Rs6, Rs7, Rs8, Rs9, Rs10;
  int Ri1, Ri2, Ri3, Ri4, Ri5, Ri6, Ri7, Ri8, Ri9, Ri10;
  long long Rll1, Rll2, Rll3, Rll4, Rll5, Rll6, Rll7, Rll8, Rll9, Rll10;
  float Rf1, Rf2, Rf3, Rf4, Rf5, Rf6, Rf7, Rf8, Rf9, Rf10;
  double Rd1, Rd2, Rd3, Rd4, Rd5, Rd6, Rd7, Rd8, Rd9, Rd10;
  long long OUT[6];
  long long EXPECTED[6];
  EXPECTED[0] = EXPECTED_1;
  EXPECTED[1] = EXPECTED_2;
  EXPECTED[2] = EXPECTED_3;
  EXPECTED[3] = EXPECTED_4;
  EXPECTED[4] = EXPECTED_5;
  EXPECTED[5] = EXPECTED_6;

  int cpuExec = 0;
  #pragma omp target map(tofrom: cpuExec)
  {
    cpuExec = omp_is_initial_device();
  }
  int gpu_threads = 512;
  int cpu_threads = 32;
  int max_threads = cpuExec ? cpu_threads : gpu_threads;

  INIT();

  //
  // Test: reduction on parallel.
  //
  for (int t = 0; t <= max_threads; t++) {
    OUT[0] = 0; OUT[1] = 0; OUT[2] = 0; OUT[3] = 0; OUT[4] = 0; OUT[5] = 0;
    int threads = t;
    TEST({
      REDUCTION_INIT();
      _Pragma("omp parallel num_threads(threads) REDUCTION_CLAUSES")
      {
        int tid = omp_get_thread_num();
        int th  = omp_get_num_threads();
        for (int i = tid; i < N; i+= th) {
          REDUCTION_BODY();
        }
      }
      REDUCTION_FINAL();
    }, VERIFY(0, 6, OUT[i], (trial+1) * EXPECTED[i]));
  }
  DUMP_SUCCESS(gpu_threads-max_threads);

  //
  // Test: reduction on multiple parallel regions.
  //
  for (int t = 1; t < 32; t++) {
    OUT[0] = 0; OUT[1] = 0; OUT[2] = 0; OUT[3] = 0; OUT[4] = 0; OUT[5] = 0;
    int threads = t;
    TEST({
      REDUCTION_INIT();
      _Pragma("omp parallel num_threads(threads) REDUCTION_CLAUSES")
      {
        int tid = omp_get_thread_num();
        int th  = omp_get_num_threads();
        for (int i = tid; i < N; i+= th) {
          REDUCTION_BODY();
        }
      }
      REDUCTION_FINAL();
      REDUCTION_INIT();
      _Pragma("omp parallel num_threads(threads+max_threads/2) REDUCTION_CLAUSES")
      {
        int tid = omp_get_thread_num();
        int th  = omp_get_num_threads();
        for (int i = tid; i < N; i+= th) {
          REDUCTION_BODY();
        }
      }
      REDUCTION_FINAL();
    }, VERIFY(0, 6, OUT[i], (trial+1) * 2*EXPECTED[i]));
  }

  //
  // Test: reduction on parallel for.
  //
  #undef CLAUSES
  #define CLAUSES REDUCTION_CLAUSES
  #include "defines-red.h"
  for (int t = 0; t <= max_threads; t++) {
    OUT[0] = 0; OUT[1] = 0; OUT[2] = 0; OUT[3] = 0; OUT[4] = 0; OUT[5] = 0;
    int threads = t;
    PARALLEL_FOR(
    {
      REDUCTION_INIT();
    },
    REDUCTION_LOOP(),
    {
      REDUCTION_FINAL();
    },
    VERIFY(0, 6, OUT[i], (trial+1) * SUMS * EXPECTED[i]))
  }
  DUMP_SUCCESS(gpu_threads-max_threads);

  //
  // Test: reduction on parallel with a nested for.
  //
  if (!cpuExec) {
    #undef CLAUSES
    #define CLAUSES REDUCTION_CLAUSES
    #include "defines-red.h"
    for (int t = 0; t <= max_threads; t++) {
      OUT[0] = 0; OUT[1] = 0; OUT[2] = 0; OUT[3] = 0; OUT[4] = 0; OUT[5] = 0;
      int threads = t;
      PARALLEL_NESTED_FOR(
      {
        REDUCTION_INIT();
      },
      REDUCTION_LOOP(),
      {
        if (omp_get_thread_num() == 0) {
          REDUCTION_FINAL();
        }
        _Pragma("omp barrier");
      },
      VERIFY(0, 6, OUT[i], (trial+1) * SUMS * EXPECTED[i]))
    }
  } else {
    //
    // Test asserts on the host runtime because the parallel region takes
    // more than 64 varargs.
    //
    DUMP_SUCCESS(gpu_threads+1);
  }

  //
  // Test: reduction on sections.
  //
  TEST({
    long long R1 = 0;
    _Pragma("omp parallel num_threads(5)")
    _Pragma("omp sections reduction(+:R1)")
    {
      _Pragma("omp section")
      R1 += All[1] + (Bll[1] + Cll[1]);
      _Pragma("omp section")
      R1 += All[10] + (Bll[10] + Cll[10]);
      _Pragma("omp section")
      R1 += All[100] + (Bll[100] + Cll[100]);
      _Pragma("omp section")
      R1 += All[20] + (Bll[20] + Cll[20]);
      _Pragma("omp section")
      R1 += All[1000] + (Bll[1000] + Cll[1000]);
    }
    OUT[0] = R1;
  }, VERIFY(0, 1, OUT[0], (5ll << 32)));

  //
  // Test: reduction on parallel sections.
  //
  TEST({
    long long R1 = 0;
    _Pragma("omp parallel sections num_threads(5) reduction(+:R1)")
    {
      _Pragma("omp section")
      R1 += All[1] + (Bll[1] + Cll[1]);
      _Pragma("omp section")
      R1 += All[10] + (Bll[10] + Cll[10]);
      _Pragma("omp section")
      R1 += All[100] + (Bll[100] + Cll[100]);
      _Pragma("omp section")
      R1 += All[20] + (Bll[20] + Cll[20]);
      _Pragma("omp section")
      R1 += All[1000] + (Bll[1000] + Cll[1000]);
    }
    OUT[0] = R1;
  }, VERIFY(0, 1, OUT[0], (5ll << 32)));

  //
  // Test: reduction on distribute parallel for.
  //
  TESTD("omp target", {
    _Pragma("omp teams num_teams(6)")
    {
      double Rd1 = 0; double Rd2 = 0;
      _Pragma("omp distribute parallel for reduction(+:Rd1) \
               reduction(-:Rd2)")
      for (int i = 0; i < N; i++) {
        Rd1 += Ad[i] + (Bd[i] + Cd[i]);
        Rd2 += Ad[i] + (Bd[i] + Cd[i]);
      }
      unsigned tm = omp_get_team_num(); // assume 6 teams
      OUT[tm] = (long long) (Rd1 + Rd2);
    }
  }, VERIFY(0, 1, OUT[0]+OUT[1]+OUT[2]+OUT[3]+OUT[4]+OUT[5],
            ( (2*N) << 16 ) ));

  //
  // Test: reduction on target parallel.
  //
  for (int t = 0; t <= max_threads; t++) {
    OUT[0] = 0; OUT[1] = 0; OUT[2] = 0; OUT[3] = 0; OUT[4] = 0; OUT[5] = 0;
    TESTD2("omp target parallel num_threads(t) REDUCTION_MAP REDUCTION_CLAUSES",
      {
        REDUCTION_INIT();
      },
      {
        int tid = omp_get_thread_num();
        int th  = omp_get_num_threads();
        for (int i = tid; i < N; i+= th) {
          REDUCTION_BODY();
        }
      },
      {
        REDUCTION_FINAL();
      },
      VERIFY(0, 6, OUT[i], (trial+1) * EXPECTED[i]));
  }
  DUMP_SUCCESS(gpu_threads-max_threads);

  //
  // Test: reduction on target parallel for.
  //
  for (int t = 0; t <= max_threads; t++) {
    OUT[0] = 0; OUT[1] = 0; OUT[2] = 0; OUT[3] = 0; OUT[4] = 0; OUT[5] = 0;
    TESTD2("omp target parallel for num_threads(t) REDUCTION_MAP REDUCTION_CLAUSES",
      {
        REDUCTION_INIT();
      },
      REDUCTION_LOOP(),
      {
        REDUCTION_FINAL();
      },
      VERIFY(0, 6, OUT[i], (trial+1) * EXPECTED[i]));
  }
  DUMP_SUCCESS(gpu_threads-max_threads);

  //
  // Test: reduction on nested parallel.
  //
  double RESULT[1024];
  int VALID[1024];
  for (int t = 32; t <= 32; t++) {
    OUT[0] = 0;
    int num_threads = t;
    int num_tests[1]; num_tests[0] = 0;
    TEST({
      _Pragma("omp parallel num_threads(num_threads)")
      {
        for (int offset = 0; offset < 32; offset++) {
          for (int factor = 1; factor < 33; factor++) {
            double Rd1 = 0; double Rd2 = 0;
            int tid = omp_get_thread_num();
            int lid = tid % 32;
            VALID[tid] = 0;

            if (lid >= offset && lid % factor == 0) {
              _Pragma("omp parallel for reduction(+:Rd1) reduction(-:Rd2)")
              for (int i = 0; i < N; i++) {
                Rd1 += Ad[i] + (Bd[i] + Cd[i]);
                Rd2 += Ad[i] + (Bd[i] + Cd[i]);
              }
              VALID[tid] = 1;
              RESULT[tid] = Rd1 + Rd2;
            }
            _Pragma("omp barrier")
            if (tid == 0) {
              for (int i = 0; i < num_threads; i++) {
                if (VALID[i]) num_tests[0]++;
                if (VALID[i] && (RESULT[i] - (double) ((2*N) << 16) > .001)) {
                  OUT[0] = 1;
                  printf ("Failed nested parallel reduction\n");
                }
              }
            }
            _Pragma("omp barrier")
          }
        }
      }
    }, VERIFY(0, 1, OUT[0] + num_tests[0], 0+(trial+1)*2156) );
  }

  //
  // Test: reduction on nested simd.
  //
  for (int t = 32; t <= 32; t++) {
    OUT[0] = 0;
    int num_threads = t;
    int num_tests[1]; num_tests[0] = 0;
    TEST({
      _Pragma("omp parallel num_threads(num_threads)")
      {
        for (int offset = 0; offset < 32; offset++) {
          for (int factor = 1; factor < 33; factor++) {
            double Rd1 = 0; double Rd2 = 0;
            int tid = omp_get_thread_num();
            int lid = tid % 32;
            VALID[tid] = 0;

            if (lid >= offset && lid % factor == 0) {
              _Pragma("omp simd reduction(+:Rd1) reduction(-:Rd2)")
              for (int i = 0; i < N; i++) {
                Rd1 += Ad[i] + (Bd[i] + Cd[i]);
                Rd2 += Ad[i] + (Bd[i] + Cd[i]);
              }
              VALID[tid] = 1;
              RESULT[tid] = Rd1 + Rd2;
            }
            _Pragma("omp barrier")
            if (tid == 0) {
              for (int i = 0; i < num_threads; i++) {
                if (VALID[i]) num_tests[0]++;
                if (VALID[i] && (RESULT[i] - (double) ((2*N) << 16) > .001)) {
                  OUT[0] = 1;
                  printf ("Failed nested simd reduction\n");
                }
              }
            }
            _Pragma("omp barrier")
          }
        }
      }
    }, VERIFY(0, 1, OUT[0] + num_tests[0], 0+(trial+1)*2156) );
  }

  double double_lb = -DBL_MAX; //-2^1023
  double double_ub = DBL_MAX; //slightly less than 2^1023

  //
  // Test: reduction from min to 1
  //
  double foo[1];
  for (int t = 0; t <= max_threads; t++) {
    TEST({
      foo[0] = double_lb;
      _Pragma("omp parallel for num_threads(t) reduction(*:foo[0])")
      for (int i=0; i<1024; i++) {
        foo[0]*=0.5;
      }
    }, VERIFY_E(0, 1, foo[0], -1.0, 0.0000001f));
  }
  DUMP_SUCCESS(gpu_threads-max_threads);

  //
  // Test: reduction from max to 1
  //
  for (int t = 0; t <= max_threads; t++) {
    TEST({
      foo[0] = double_ub;
      _Pragma("omp parallel for num_threads(t) reduction(*:foo[0])")
      for (int i=0; i<1024; i++) {
        foo[0]*=0.5;
      }
    }, VERIFY_E(0, 1, foo[0], 1.0, 0.0000001f));
  }
  DUMP_SUCCESS(gpu_threads-max_threads);

  float float_lb = -FLT_MAX;
  float float_ub = FLT_MAX;

  //
  // Test: reduction from min to 1
  //
  float bar[1];
  for (int t = 0; t <= max_threads; t++) {
    TEST({
      bar[0] = float_lb;
      _Pragma("omp parallel for num_threads(t) reduction(*:bar[0])")
      for (int i=0; i<128; i++) {
        bar[0]*=0.5;
      }
    }, VERIFY_E(0, 1, bar[0], -1.0f, 0.0000001f));
  }
  DUMP_SUCCESS(gpu_threads-max_threads);

  //
  // Test: reduction from max to 1
  //
  for (int t = 0; t <= max_threads; t++) {
    TEST({
      bar[0] = float_ub;
      _Pragma("omp parallel for num_threads(t) reduction(*:bar[0])")
      for (int i=0; i<128; i++) {
        bar[0]*=0.5;
      }
    }, VERIFY_E(0, 1, bar[0], 1.0f, 0.0000001f));
  }
  DUMP_SUCCESS(gpu_threads-max_threads);

  return 0;
}


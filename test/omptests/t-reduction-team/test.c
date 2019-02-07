
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

#define EXPECTED_RESULT ( \
INIT1 + INIT2 + \
(N << 16) + (N << 16) + \
/* + (2*(N-1)+1) - (N-1) */ + \
(INITd5*2*2*2) + \
1 + 1 \
)

#define REDUCTION_CLAUSES reduction(+:Rd1) reduction(-:Rd2) reduction(*:Rd5) \
                  reduction(&&:Rd9) reduction(||:Rd10)
//reduction(max:Ri3) reduction(min:Ri4)

#define REDUCTION_MAP map(tofrom: Rd1, Rd2, Rd5, Rd9, Rd10)

#define REDUCTION_INIT() {           \
      Rd1 = INIT1; Rd2 = INIT2;      \
      Rd3 = INIT3; Rd4 = INIT4;      \
      Rd5 = INITd5; Rd6 = INITd6;    \
      Rd7 = INIT7; Rd8 = INIT8;      \
      Rd9 = INIT9; Rd10 = INIT10;    \
}

#define REDUCTION_BODY() \
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
      OUT[0] += (long long) (Rd1 + Rd2 /*+ Rd3 + Rd4 */ + Rd5 + Rd9 + Rd10); \
}

int main(void) {
  check_offloading();

  double Ad[N], Bd[N], Cd[N], Dd[N], Ed[N];
  double Rd1, Rd2, Rd3, Rd4, Rd5, Rd6, Rd7, Rd8, Rd9, Rd10;
  long long OUT[1];
  long long EXPECTED[1];
  EXPECTED[0] = EXPECTED_RESULT;

  int cpuExec = 0;
  #pragma omp target map(tofrom: cpuExec)
  {
    cpuExec = omp_is_initial_device();
  }
  int gpu_threads = 512;
  int cpu_threads = 32;
  int max_threads = cpuExec ? cpu_threads : gpu_threads;

  INIT();

  if (cpuExec) {
    // Certain tests in this testcase fails on the host.  A bug report has
    // been filed: https://puna0.watson.ibm.com/T143
    // Disabling this test on the host for now.
    DUMP_SUCCESS(3153);
    return 0;
  }

  //
  // Test: reduction on teams.
  //
  for (int tms = 1 ; tms <= 512 ; tms *= 7) {
    OUT[0] = 0;
    TESTD2("omp target REDUCTION_MAP",
      {
        REDUCTION_INIT();
      },
      {
      _Pragma("omp teams num_teams(tms) REDUCTION_CLAUSES")
      {
        int tid = omp_get_team_num();
        int th  = omp_get_num_teams();
        for (int i = tid; i < N; i+= th) {
          REDUCTION_BODY();
        }
      }
      },
      {
        REDUCTION_FINAL();
      }, VERIFY(0, 1, OUT[i], (trial+1) * EXPECTED[i]));
  }

  //
  // Test: reduction on teams with nested parallel.
  //
  for (int tms = 1 ; tms <= 512 ; tms *= 7) {
    OUT[0] = 0;
    TESTD2("omp target REDUCTION_MAP",
      {
        REDUCTION_INIT();
      },
      {
      _Pragma("omp teams num_teams(tms) REDUCTION_CLAUSES")
      {
        int tid = omp_get_team_num();
        int th  = omp_get_num_teams();
        _Pragma("omp parallel for REDUCTION_CLAUSES")
        for (int i = tid; i < N; i+= th) {
          REDUCTION_BODY();
        }
      }
      },
      {
        REDUCTION_FINAL();
      }, VERIFY(0, 1, OUT[i], (trial+1) * EXPECTED[i]));
  }

  //
  // Test: reduction on target teams.
  //
  for (int tms = 1 ; tms <= 512 ; tms *= 7) {
    OUT[0] = 0;
    TESTD2("omp target teams num_teams(tms) REDUCTION_MAP REDUCTION_CLAUSES",
      {
        REDUCTION_INIT();
      },
      {
      {
        int tid = omp_get_team_num();
        int th  = omp_get_num_teams();
        for (int i = tid; i < N; i+= th) {
          REDUCTION_BODY();
        }
      }
      },
      {
        REDUCTION_FINAL();
      }, VERIFY(0, 1, OUT[i], (trial+1) * EXPECTED[i]));
  }

  //
  // Test: reduction on teams with nested parallel.
  //
  for (int tms = 1 ; tms <= 512 ; tms *= 7) {
    OUT[0] = 0;
    TESTD2("omp target teams num_teams(tms) REDUCTION_MAP REDUCTION_CLAUSES",
      {
        REDUCTION_INIT();
      },
      {
      {
        int tid = omp_get_team_num();
        int th  = omp_get_num_teams();
        _Pragma("omp parallel for REDUCTION_CLAUSES")
        for (int i = tid; i < N; i+= th) {
          REDUCTION_BODY();
        }
      }
      },
      {
        REDUCTION_FINAL();
      }, VERIFY(0, 1, OUT[i], (trial+1) * EXPECTED[i]));
  }

  //
  // Test: reduction on target parallel with nested parallel.
  //
  OUT[0] = 0;
  TESTD2("omp target parallel num_threads(30) REDUCTION_MAP REDUCTION_CLAUSES",
    {
      REDUCTION_INIT();
    },
    {
    {
      int tid = omp_get_thread_num();
      int th  = omp_get_num_threads();
      _Pragma("omp simd REDUCTION_CLAUSES")
      for (int i = tid; i < N; i+= th) {
        REDUCTION_BODY();
      }
    }
    },
    {
      REDUCTION_FINAL();
    }, VERIFY(0, 1, OUT[i], (trial+1) * EXPECTED[i]));

  //
  // Test: reduction on target teams distribute parallel for.
  //
  for (int tms = 1 ; tms <= 512 ; tms *= 7) {
    for (int ths = 1 ; ths <= 1024 ; ths *= 9) {
      OUT[0] = 0;
      TESTD2("omp target teams distribute parallel for num_teams(tms) thread_limit(ths) REDUCTION_MAP REDUCTION_CLAUSES",
        {
          REDUCTION_INIT();
        },
        REDUCTION_LOOP(),
        {
          REDUCTION_FINAL();
        },
        VERIFY(0, 1, OUT[i], (trial+1) * EXPECTED[i]));
    }
  }

  //
  // Test: reduction on target teams distribute parallel for with schedule static nochunk.
  //
  for (int tms = 1 ; tms <= 512 ; tms *= 7) {
    for (int ths = 1 ; ths <= 1024 ; ths *= 9) {
      OUT[0] = 0;
      TESTD2("omp target teams distribute parallel for num_teams(tms) thread_limit(ths) schedule(static) REDUCTION_MAP REDUCTION_CLAUSES",
        {
          REDUCTION_INIT();
        },
        REDUCTION_LOOP(),
        {
          REDUCTION_FINAL();
        },
        VERIFY(0, 1, OUT[i], (trial+1) * EXPECTED[i]));
    }
  }

  //
  // Test: reduction on target teams distribute parallel for with schedule static chunk.
  //
  for (int tms = 1 ; tms <= 512 ; tms *= 7) {
    for (int ths = 1 ; ths <= 1024 ; ths *= 9) {
      for(int sch = 1 ; sch <= N ; sch *= 9) {
        OUT[0] = 0;
        TESTD2("omp target teams distribute parallel for num_teams(tms) thread_limit(ths) schedule(static,sch) REDUCTION_MAP REDUCTION_CLAUSES",
          {
            REDUCTION_INIT();
          },
          REDUCTION_LOOP(),
          {
            REDUCTION_FINAL();
          },
          VERIFY(0, 1, OUT[i], (trial+1) * EXPECTED[i]));
      }
    }
  }

  //
  // Test: reduction on target teams distribute parallel for with schedule dynamic nochunk.
  //
  for (int tms = 1 ; tms <= 512 ; tms *= 7) {
    for (int ths = 1 ; ths <= 1024 ; ths *= 9) {
      OUT[0] = 0;
      TESTD2("omp target teams distribute parallel for num_teams(tms) thread_limit(ths) schedule(dynamic) REDUCTION_MAP REDUCTION_CLAUSES",
        {
          REDUCTION_INIT();
        },
        REDUCTION_LOOP(),
        {
          REDUCTION_FINAL();
        },
        VERIFY(0, 1, OUT[i], (trial+1) * EXPECTED[i]));
    }
  }

  //
  // Test: reduction on target teams distribute parallel for with schedule dynamic chunk.
  //
  for (int tms = 1 ; tms <= 512 ; tms *= 7) {
    for (int ths = 1 ; ths <= 1024 ; ths *= 9) {
      for(int sch = 1 ; sch <= N ; sch *= 9) {
        OUT[0] = 0;
        TESTD2("omp target teams distribute parallel for num_teams(tms) thread_limit(ths) schedule(dynamic,sch) REDUCTION_MAP REDUCTION_CLAUSES",
          {
            REDUCTION_INIT();
          },
          REDUCTION_LOOP(),
          {
            REDUCTION_FINAL();
          },
          VERIFY(0, 1, OUT[i], (trial+1) * EXPECTED[i]));
      }
    }
  }

  //
  // Test: reduction on target teams distribute parallel for with dist_schedule static nochunk.
  //
  for (int tms = 1 ; tms <= 512 ; tms *= 7) {
    for (int ths = 1 ; ths <= 1024 ; ths *= 9) {
      OUT[0] = 0;
      TESTD2("omp target teams distribute parallel for num_teams(tms) thread_limit(ths) dist_schedule(static) REDUCTION_MAP REDUCTION_CLAUSES",
        {
          REDUCTION_INIT();
        },
        REDUCTION_LOOP(),
        {
          REDUCTION_FINAL();
        },
        VERIFY(0, 1, OUT[i], (trial+1) * EXPECTED[i]));
    }
  }

  //
  // Test: reduction on target teams distribute parallel for with dist_schedule static nochunk, schedule static nochunk.
  //
  for (int tms = 1 ; tms <= 512 ; tms *= 7) {
    for (int ths = 1 ; ths <= 1024 ; ths *= 9) {
      OUT[0] = 0;
      TESTD2("omp target teams distribute parallel for num_teams(tms) thread_limit(ths) dist_schedule(static) schedule(static) REDUCTION_MAP REDUCTION_CLAUSES",
        {
          REDUCTION_INIT();
        },
        REDUCTION_LOOP(),
        {
          REDUCTION_FINAL();
        },
        VERIFY(0, 1, OUT[i], (trial+1) * EXPECTED[i]));
    }
  }

  //
  // Test: reduction on target teams distribute parallel for with dist_schedule static nochunk, schedule static chunk.
  //
  for (int tms = 1 ; tms <= 512 ; tms *= 7) {
    for (int ths = 1 ; ths <= 1024 ; ths *= 9) {
      for(int sch = 1 ; sch <= N ; sch *= 9) {
        OUT[0] = 0;
        TESTD2("omp target teams distribute parallel for num_teams(tms) thread_limit(ths) dist_schedule(static) schedule(static,sch) REDUCTION_MAP REDUCTION_CLAUSES",
          {
            REDUCTION_INIT();
          },
          REDUCTION_LOOP(),
          {
            REDUCTION_FINAL();
          },
          VERIFY(0, 1, OUT[i], (trial+1) * EXPECTED[i]));
      }
    }
  }

  //
  // Test: reduction on target teams distribute parallel for with dist_schedule static chunk, schedule static nochunk.
  //
  for (int tms = 1 ; tms <= 512 ; tms *= 7) {
    for (int ths = 1 ; ths <= 1024 ; ths *= 9) {
      for(int sch = 1 ; sch <= N ; sch *= 9) {
        OUT[0] = 0;
        TESTD2("omp target teams distribute parallel for num_teams(tms) thread_limit(ths) dist_schedule(static,sch) schedule(static) REDUCTION_MAP REDUCTION_CLAUSES",
          {
            REDUCTION_INIT();
          },
          REDUCTION_LOOP(),
          {
            REDUCTION_FINAL();
          },
          VERIFY(0, 1, OUT[i], (trial+1) * EXPECTED[i]));
      }
    }
  }

  //
  // Test: reduction on target teams distribute parallel for with dist_schedule static chunk, schedule static chunk.
  //
  for (int tms = 1 ; tms <= 512 ; tms *= 7) {
    for (int ths = 1 ; ths <= 1024 ; ths *= 9) {
      for(int sch = 1 ; sch <= N ; sch *= 9) {
        OUT[0] = 0;
        TESTD2("omp target teams distribute parallel for num_teams(tms) thread_limit(ths) dist_schedule(static,sch) schedule(static,sch) REDUCTION_MAP REDUCTION_CLAUSES",
          {
            REDUCTION_INIT();
          },
          REDUCTION_LOOP(),
          {
            REDUCTION_FINAL();
          },
          VERIFY(0, 1, OUT[i], (trial+1) * EXPECTED[i]));
      }
    }
  }

  //
  // Test: reduction on target teams distribute parallel for with dist_schedule static chunk, schedule dynamic nochunk.
  //
  for (int tms = 1 ; tms <= 512 ; tms *= 7) {
    for (int ths = 1 ; ths <= 1024 ; ths *= 9) {
      for(int sch = 1 ; sch <= N ; sch *= 9) {
        OUT[0] = 0;
        TESTD2("omp target teams distribute parallel for num_teams(tms) thread_limit(ths) dist_schedule(static,sch) schedule(dynamic) REDUCTION_MAP REDUCTION_CLAUSES",
          {
            REDUCTION_INIT();
          },
          REDUCTION_LOOP(),
          {
            REDUCTION_FINAL();
          },
          VERIFY(0, 1, OUT[i], (trial+1) * EXPECTED[i]));
      }
    }
  }

  //
  // Test: reduction on target teams distribute parallel for with dist_schedule static chunk,schedule dynamic chunk.
  //
  for (int tms = 1 ; tms <= 512 ; tms *= 7) {
    for (int ths = 1 ; ths <= 1024 ; ths *= 9) {
      for(int sch = 1 ; sch <= N ; sch *= 9) {
        OUT[0] = 0;
        TESTD2("omp target teams distribute parallel for num_teams(tms) thread_limit(ths) dist_schedule(static,sch) schedule(dynamic,sch) REDUCTION_MAP REDUCTION_CLAUSES",
          {
            REDUCTION_INIT();
          },
          REDUCTION_LOOP(),
          {
            REDUCTION_FINAL();
          },
          VERIFY(0, 1, OUT[i], (trial+1) * EXPECTED[i]));
      }
    }
  }

  //
  // Test: reduction on target teams distribute parallel for with dist_schedule static chunk, schedule guided nochunk.
  //
  for (int tms = 1 ; tms <= 512 ; tms *= 7) {
    for (int ths = 1 ; ths <= 1024 ; ths *= 9) {
      for(int sch = 1 ; sch <= N ; sch *= 9) {
        OUT[0] = 0;
        TESTD2("omp target teams distribute parallel for num_teams(tms) thread_limit(ths) dist_schedule(static,sch) schedule(guided) REDUCTION_MAP REDUCTION_CLAUSES",
          {
            REDUCTION_INIT();
          },
          REDUCTION_LOOP(),
          {
            REDUCTION_FINAL();
          },
          VERIFY(0, 1, OUT[i], (trial+1) * EXPECTED[i]));
      }
    }
  }

  //
  // Test: reduction on target teams distribute parallel for with dist_schedule static chunk, schedule guided chunk.
  //
  for (int tms = 1 ; tms <= 512 ; tms *= 7) {
    for (int ths = 1 ; ths <= 1024 ; ths *= 9) {
      for(int sch = 1 ; sch <= N ; sch *= 9) {
        OUT[0] = 0;
        TESTD2("omp target teams distribute parallel for num_teams(tms) thread_limit(ths) dist_schedule(static,sch) schedule(guided,sch) REDUCTION_MAP REDUCTION_CLAUSES",
          {
            REDUCTION_INIT();
          },
          REDUCTION_LOOP(),
          {
            REDUCTION_FINAL();
          },
          VERIFY(0, 1, OUT[i], (trial+1) * EXPECTED[i]));
      }
    }
  }

  //
  // Test: reduction on target teams distribute parallel for with dist_schedule static chunk, schedule auto.
  //
  for (int tms = 1 ; tms <= 512 ; tms *= 7) {
    for (int ths = 1 ; ths <= 1024 ; ths *= 9) {
      for(int sch = 1 ; sch <= N ; sch *= 9) {
        OUT[0] = 0;
        TESTD2("omp target teams distribute parallel for num_teams(tms) thread_limit(ths) dist_schedule(static,sch) schedule(dynamic) REDUCTION_MAP REDUCTION_CLAUSES",
          {
            REDUCTION_INIT();
          },
          REDUCTION_LOOP(),
          {
            REDUCTION_FINAL();
          },
          VERIFY(0, 1, OUT[i], (trial+1) * EXPECTED[i]));
      }
    }
  }

  //
  // Test: reduction on target teams distribute parallel for with dist_schedule static chunk, schedule runtime.
  //
  for (int tms = 1 ; tms <= 512 ; tms *= 7) {
    for (int ths = 1 ; ths <= 1024 ; ths *= 9) {
      for(int sch = 1 ; sch <= N ; sch *= 9) {
        OUT[0] = 0;
        TESTD2("omp target teams distribute parallel for num_teams(tms) thread_limit(ths) dist_schedule(static,sch) schedule(runtime) REDUCTION_MAP REDUCTION_CLAUSES",
          {
            REDUCTION_INIT();
          },
          REDUCTION_LOOP(),
          {
            REDUCTION_FINAL();
          },
          VERIFY(0, 1, OUT[i], (trial+1) * EXPECTED[i]));
      }
    }
  }

  //
  // Test: reduction on teams distribute parallel for.
  //
  for (int tms = 1 ; tms <= 512 ; tms *= 7) {
    for (int ths = 1 ; ths <= 1024 ; ths *= 9) {
      OUT[0] = 0;
      TESTD2("omp target REDUCTION_MAP",
        {
          REDUCTION_INIT();
        },
        {
        _Pragma("teams distribute parallel for num_teams(tms) thread_limit(ths) REDUCTION_CLAUSES")
        REDUCTION_LOOP()
        },
        {
          REDUCTION_FINAL();
        },
        VERIFY(0, 1, OUT[i], (trial+1) * EXPECTED[i]));
    }
  }

  //
  // Test: reduction on teams distribute parallel for with schedule static nochunk.
  //
  for (int tms = 1 ; tms <= 512 ; tms *= 7) {
    for (int ths = 1 ; ths <= 1024 ; ths *= 9) {
      OUT[0] = 0;
      TESTD2("omp target REDUCTION_MAP",
        {
          REDUCTION_INIT();
        },
        {
        _Pragma("teams distribute parallel for num_teams(tms) thread_limit(ths) schedule(static) REDUCTION_CLAUSES")
        REDUCTION_LOOP()
        },
        {
          REDUCTION_FINAL();
        },
        VERIFY(0, 1, OUT[i], (trial+1) * EXPECTED[i]));
    }
  }

  //
  // Test: reduction on teams distribute parallel for with schedule static chunk.
  //
  for (int tms = 1 ; tms <= 512 ; tms *= 7) {
    for (int ths = 1 ; ths <= 1024 ; ths *= 9) {
      for(int sch = 1 ; sch <= N ; sch *= 9) {
        OUT[0] = 0;
        TESTD2("omp target REDUCTION_MAP",
          {
            REDUCTION_INIT();
          },
          {
          _Pragma("teams distribute parallel for num_teams(tms) thread_limit(ths) schedule(static,sch) REDUCTION_CLAUSES")
          REDUCTION_LOOP()
          },
          {
            REDUCTION_FINAL();
          },
          VERIFY(0, 1, OUT[i], (trial+1) * EXPECTED[i]));
      }
    }
  }

  //
  // Test: reduction on teams distribute parallel for with schedule dynamic nochunk.
  //
  for (int tms = 1 ; tms <= 512 ; tms *= 7) {
    for (int ths = 1 ; ths <= 1024 ; ths *= 9) {
      OUT[0] = 0;
      TESTD2("omp target REDUCTION_MAP",
        {
          REDUCTION_INIT();
        },
        {
        _Pragma("teams distribute parallel for num_teams(tms) thread_limit(ths) schedule(dynamic) REDUCTION_CLAUSES")
        REDUCTION_LOOP()
        },
        {
          REDUCTION_FINAL();
        },
        VERIFY(0, 1, OUT[i], (trial+1) * EXPECTED[i]));
    }
  }

  //
  // Test: reduction on teams distribute parallel for with schedule dynamic chunk.
  //
  for (int tms = 1 ; tms <= 512 ; tms *= 7) {
    for (int ths = 1 ; ths <= 1024 ; ths *= 9) {
      for(int sch = 1 ; sch <= N ; sch *= 9) {
        OUT[0] = 0;
        TESTD2("omp target REDUCTION_MAP",
          {
            REDUCTION_INIT();
          },
          {
          _Pragma("teams distribute parallel for num_teams(tms) thread_limit(ths) schedule(dynamic,sch) REDUCTION_CLAUSES")
          REDUCTION_LOOP()
          },
          {
            REDUCTION_FINAL();
          },
          VERIFY(0, 1, OUT[i], (trial+1) * EXPECTED[i]));
      }
    }
  }

  //
  // Test: reduction on teams distribute parallel for with dist_schedule static nochunk.
  //
  for (int tms = 1 ; tms <= 512 ; tms *= 7) {
    for (int ths = 1 ; ths <= 1024 ; ths *= 9) {
      OUT[0] = 0;
      TESTD2("omp target REDUCTION_MAP",
        {
          REDUCTION_INIT();
        },
        {
        _Pragma("teams distribute parallel for num_teams(tms) thread_limit(ths) dist_schedule(static) REDUCTION_CLAUSES")
        REDUCTION_LOOP()
        },
        {
          REDUCTION_FINAL();
        },
        VERIFY(0, 1, OUT[i], (trial+1) * EXPECTED[i]));
    }
  }

  //
  // Test: reduction on teams distribute parallel for with dist_schedule static nochunk, schedule static nochunk.
  //
  for (int tms = 1 ; tms <= 512 ; tms *= 7) {
    for (int ths = 1 ; ths <= 1024 ; ths *= 9) {
      OUT[0] = 0;
      TESTD2("omp target REDUCTION_MAP",
        {
          REDUCTION_INIT();
        },
        {
        _Pragma("teams distribute parallel for num_teams(tms) thread_limit(ths) dist_schedule(static) schedule(static) REDUCTION_CLAUSES")
        REDUCTION_LOOP()
        },
        {
          REDUCTION_FINAL();
        },
        VERIFY(0, 1, OUT[i], (trial+1) * EXPECTED[i]));
    }
  }

  //
  // Test: reduction on teams distribute parallel for with dist_schedule static nochunk, schedule static chunk.
  //
  for (int tms = 1 ; tms <= 512 ; tms *= 7) {
    for (int ths = 1 ; ths <= 1024 ; ths *= 9) {
      for(int sch = 1 ; sch <= N ; sch *= 9) {
        OUT[0] = 0;
        TESTD2("omp target REDUCTION_MAP",
          {
            REDUCTION_INIT();
          },
          {
          _Pragma("teams distribute parallel for num_teams(tms) thread_limit(ths) dist_schedule(static) schedule(static,sch) REDUCTION_CLAUSES")
          REDUCTION_LOOP()
          },
          {
            REDUCTION_FINAL();
          },
          VERIFY(0, 1, OUT[i], (trial+1) * EXPECTED[i]));
      }
    }
  }

  //
  // Test: reduction on teams distribute parallel for with dist_schedule static chunk, schedule static nochunk.
  //
  for (int tms = 1 ; tms <= 512 ; tms *= 7) {
    for (int ths = 1 ; ths <= 1024 ; ths *= 9) {
      for(int sch = 1 ; sch <= N ; sch *= 9) {
        OUT[0] = 0;
        TESTD2("omp target REDUCTION_MAP",
          {
            REDUCTION_INIT();
          },
          {
          _Pragma("teams distribute parallel for num_teams(tms) thread_limit(ths) dist_schedule(static,sch) schedule(static) REDUCTION_CLAUSES")
          REDUCTION_LOOP()
          },
          {
            REDUCTION_FINAL();
          },
          VERIFY(0, 1, OUT[i], (trial+1) * EXPECTED[i]));
      }
    }
  }

  //
  // Test: reduction on teams distribute parallel for with dist_schedule static chunk, schedule static chunk.
  //
  for (int tms = 1 ; tms <= 512 ; tms *= 7) {
    for (int ths = 1 ; ths <= 1024 ; ths *= 9) {
      for(int sch = 1 ; sch <= N ; sch *= 9) {
        OUT[0] = 0;
        TESTD2("omp target REDUCTION_MAP",
          {
            REDUCTION_INIT();
          },
          {
          _Pragma("teams distribute parallel for num_teams(tms) thread_limit(ths) dist_schedule(static,sch) schedule(static,sch) REDUCTION_CLAUSES")
          REDUCTION_LOOP()
          },
          {
            REDUCTION_FINAL();
          },
          VERIFY(0, 1, OUT[i], (trial+1) * EXPECTED[i]));
      }
    }
  }

  //
  // Test: reduction on teams distribute parallel for with dist_schedule static chunk, schedule dynamic nochunk.
  //
  for (int tms = 1 ; tms <= 512 ; tms *= 7) {
    for (int ths = 1 ; ths <= 1024 ; ths *= 9) {
      for(int sch = 1 ; sch <= N ; sch *= 9) {
        OUT[0] = 0;
        TESTD2("omp target REDUCTION_MAP",
          {
            REDUCTION_INIT();
          },
          {
          _Pragma("teams distribute parallel for num_teams(tms) thread_limit(ths) dist_schedule(static,sch) schedule(dynamic) REDUCTION_CLAUSES")
          REDUCTION_LOOP()
          },
          {
            REDUCTION_FINAL();
          },
          VERIFY(0, 1, OUT[i], (trial+1) * EXPECTED[i]));
      }
    }
  }

  //
  // Test: reduction on teams distribute parallel for with dist_schedule static chunk,schedule dynamic chunk.
  //
  for (int tms = 1 ; tms <= 512 ; tms *= 7) {
    for (int ths = 1 ; ths <= 1024 ; ths *= 9) {
      for(int sch = 1 ; sch <= N ; sch *= 9) {
        OUT[0] = 0;
        TESTD2("omp target REDUCTION_MAP",
          {
            REDUCTION_INIT();
          },
          {
          _Pragma("teams distribute parallel for num_teams(tms) thread_limit(ths) dist_schedule(static,sch) schedule(dynamic,sch) REDUCTION_CLAUSES")
          REDUCTION_LOOP()
          },
          {
            REDUCTION_FINAL();
          },
          VERIFY(0, 1, OUT[i], (trial+1) * EXPECTED[i]));
      }
    }
  }

  //
  // Test: reduction on teams distribute parallel for with dist_schedule static chunk, schedule guided nochunk.
  //
  for (int tms = 1 ; tms <= 512 ; tms *= 7) {
    for (int ths = 1 ; ths <= 1024 ; ths *= 9) {
      for(int sch = 1 ; sch <= N ; sch *= 9) {
        OUT[0] = 0;
        TESTD2("omp target REDUCTION_MAP",
          {
            REDUCTION_INIT();
          },
          {
          _Pragma("teams distribute parallel for num_teams(tms) thread_limit(ths) dist_schedule(static,sch) schedule(guided) REDUCTION_CLAUSES")
          REDUCTION_LOOP()
          },
          {
            REDUCTION_FINAL();
          },
          VERIFY(0, 1, OUT[i], (trial+1) * EXPECTED[i]));
      }
    }
  }

  //
  // Test: reduction on teams distribute parallel for with dist_schedule static chunk, schedule guided chunk.
  //
  for (int tms = 1 ; tms <= 512 ; tms *= 7) {
    for (int ths = 1 ; ths <= 1024 ; ths *= 9) {
      for(int sch = 1 ; sch <= N ; sch *= 9) {
        OUT[0] = 0;
        TESTD2("omp target REDUCTION_MAP",
          {
            REDUCTION_INIT();
          },
          {
          _Pragma("teams distribute parallel for num_teams(tms) thread_limit(ths) dist_schedule(static,sch) schedule(guided,sch) REDUCTION_CLAUSES")
          REDUCTION_LOOP()
          },
          {
            REDUCTION_FINAL();
          },
          VERIFY(0, 1, OUT[i], (trial+1) * EXPECTED[i]));
      }
    }
  }

  //
  // Test: reduction on teams distribute parallel for with dist_schedule static chunk, schedule auto.
  //
  for (int tms = 1 ; tms <= 512 ; tms *= 7) {
    for (int ths = 1 ; ths <= 1024 ; ths *= 9) {
      for(int sch = 1 ; sch <= N ; sch *= 9) {
        OUT[0] = 0;
        TESTD2("omp target REDUCTION_MAP",
          {
            REDUCTION_INIT();
          },
          {
          _Pragma("teams distribute parallel for num_teams(tms) thread_limit(ths) dist_schedule(static,sch) schedule(dynamic) REDUCTION_CLAUSES")
          REDUCTION_LOOP()
          },
          {
            REDUCTION_FINAL();
          },
          VERIFY(0, 1, OUT[i], (trial+1) * EXPECTED[i]));
      }
    }
  }

  //
  // Test: reduction on teams distribute parallel for with dist_schedule static chunk, schedule runtime.
  //
  for (int tms = 1 ; tms <= 512 ; tms *= 7) {
    for (int ths = 1 ; ths <= 1024 ; ths *= 9) {
      for(int sch = 1 ; sch <= N ; sch *= 9) {
        OUT[0] = 0;
        TESTD2("omp target REDUCTION_MAP",
          {
            REDUCTION_INIT();
          },
          {
          _Pragma("teams distribute parallel for num_teams(tms) thread_limit(ths) dist_schedule(static,sch) schedule(runtime) REDUCTION_CLAUSES")
          REDUCTION_LOOP()
          },
          {
            REDUCTION_FINAL();
          },
          VERIFY(0, 1, OUT[i], (trial+1) * EXPECTED[i]));
      }
    }
  }

  //
  // Test: reduction on target teams distribute parallel for simd.
  //
  for (int tms = 1 ; tms <= 512 ; tms *= 7) {
    for (int ths = 1 ; ths <= 1024 ; ths *= 9) {
      OUT[0] = 0;
      TESTD2("omp target teams distribute parallel for simd num_teams(tms) thread_limit(ths) REDUCTION_MAP REDUCTION_CLAUSES",
        {
          REDUCTION_INIT();
        },
        REDUCTION_LOOP(),
        {
          REDUCTION_FINAL();
        },
        VERIFY(0, 1, OUT[i], (trial+1) * EXPECTED[i]));
    }
  }

  //
  // Test: reduction on target teams distribute parallel for simd with schedule static nochunk.
  //
  for (int tms = 1 ; tms <= 512 ; tms *= 7) {
    for (int ths = 1 ; ths <= 1024 ; ths *= 9) {
      OUT[0] = 0;
      TESTD2("omp target teams distribute parallel for simd num_teams(tms) thread_limit(ths) schedule(static) REDUCTION_MAP REDUCTION_CLAUSES",
        {
          REDUCTION_INIT();
        },
        REDUCTION_LOOP(),
        {
          REDUCTION_FINAL();
        },
        VERIFY(0, 1, OUT[i], (trial+1) * EXPECTED[i]));
    }
  }

  //
  // Test: reduction on target teams distribute parallel for simd with schedule static chunk.
  //
  for (int tms = 1 ; tms <= 512 ; tms *= 7) {
    for (int ths = 1 ; ths <= 1024 ; ths *= 9) {
      for(int sch = 1 ; sch <= N ; sch *= 9) {
        OUT[0] = 0;
        TESTD2("omp target teams distribute parallel for simd num_teams(tms) thread_limit(ths) schedule(static,sch) REDUCTION_MAP REDUCTION_CLAUSES",
          {
            REDUCTION_INIT();
          },
          REDUCTION_LOOP(),
          {
            REDUCTION_FINAL();
          },
          VERIFY(0, 1, OUT[i], (trial+1) * EXPECTED[i]));
      }
    }
  }

  //
  // Test: reduction on target teams distribute parallel for simd with schedule dynamic nochunk.
  //
  for (int tms = 1 ; tms <= 512 ; tms *= 7) {
    for (int ths = 1 ; ths <= 1024 ; ths *= 9) {
      OUT[0] = 0;
      TESTD2("omp target teams distribute parallel for simd num_teams(tms) thread_limit(ths) schedule(dynamic) REDUCTION_MAP REDUCTION_CLAUSES",
        {
          REDUCTION_INIT();
        },
        REDUCTION_LOOP(),
        {
          REDUCTION_FINAL();
        },
        VERIFY(0, 1, OUT[i], (trial+1) * EXPECTED[i]));
    }
  }

  //
  // Test: reduction on target teams distribute parallel for simd with schedule dynamic chunk.
  //
  for (int tms = 1 ; tms <= 512 ; tms *= 7) {
    for (int ths = 1 ; ths <= 1024 ; ths *= 9) {
      for(int sch = 1 ; sch <= N ; sch *= 9) {
        OUT[0] = 0;
        TESTD2("omp target teams distribute parallel for simd num_teams(tms) thread_limit(ths) schedule(dynamic,sch) REDUCTION_MAP REDUCTION_CLAUSES",
          {
            REDUCTION_INIT();
          },
          REDUCTION_LOOP(),
          {
            REDUCTION_FINAL();
          },
          VERIFY(0, 1, OUT[i], (trial+1) * EXPECTED[i]));
      }
    }
  }

  //
  // Test: reduction on target teams distribute parallel for simd with dist_schedule static nochunk.
  //
  for (int tms = 1 ; tms <= 512 ; tms *= 7) {
    for (int ths = 1 ; ths <= 1024 ; ths *= 9) {
      OUT[0] = 0;
      TESTD2("omp target teams distribute parallel for simd num_teams(tms) thread_limit(ths) dist_schedule(static) REDUCTION_MAP REDUCTION_CLAUSES",
        {
          REDUCTION_INIT();
        },
        REDUCTION_LOOP(),
        {
          REDUCTION_FINAL();
        },
        VERIFY(0, 1, OUT[i], (trial+1) * EXPECTED[i]));
    }
  }

  //
  // Test: reduction on target teams distribute parallel for simd with dist_schedule static nochunk, schedule static nochunk.
  //
  for (int tms = 1 ; tms <= 512 ; tms *= 7) {
    for (int ths = 1 ; ths <= 1024 ; ths *= 9) {
      OUT[0] = 0;
      TESTD2("omp target teams distribute parallel for simd num_teams(tms) thread_limit(ths) dist_schedule(static) schedule(static) REDUCTION_MAP REDUCTION_CLAUSES",
        {
          REDUCTION_INIT();
        },
        REDUCTION_LOOP(),
        {
          REDUCTION_FINAL();
        },
        VERIFY(0, 1, OUT[i], (trial+1) * EXPECTED[i]));
    }
  }

  //
  // Test: reduction on target teams distribute parallel for simd with dist_schedule static nochunk, schedule static chunk.
  //
  for (int tms = 1 ; tms <= 512 ; tms *= 7) {
    for (int ths = 1 ; ths <= 1024 ; ths *= 9) {
      for(int sch = 1 ; sch <= N ; sch *= 9) {
        OUT[0] = 0;
        TESTD2("omp target teams distribute parallel for simd num_teams(tms) thread_limit(ths) dist_schedule(static) schedule(static,sch) REDUCTION_MAP REDUCTION_CLAUSES",
          {
            REDUCTION_INIT();
          },
          REDUCTION_LOOP(),
          {
            REDUCTION_FINAL();
          },
          VERIFY(0, 1, OUT[i], (trial+1) * EXPECTED[i]));
      }
    }
  }

  //
  // Test: reduction on target teams distribute parallel for simd with dist_schedule static chunk, schedule static nochunk.
  //
  for (int tms = 1 ; tms <= 512 ; tms *= 7) {
    for (int ths = 1 ; ths <= 1024 ; ths *= 9) {
      for(int sch = 1 ; sch <= N ; sch *= 9) {
        OUT[0] = 0;
        TESTD2("omp target teams distribute parallel for simd num_teams(tms) thread_limit(ths) dist_schedule(static,sch) schedule(static) REDUCTION_MAP REDUCTION_CLAUSES",
          {
            REDUCTION_INIT();
          },
          REDUCTION_LOOP(),
          {
            REDUCTION_FINAL();
          },
          VERIFY(0, 1, OUT[i], (trial+1) * EXPECTED[i]));
      }
    }
  }

  //
  // Test: reduction on target teams distribute parallel for simd with dist_schedule static chunk, schedule static chunk.
  //
  for (int tms = 1 ; tms <= 512 ; tms *= 7) {
    for (int ths = 1 ; ths <= 1024 ; ths *= 9) {
      for(int sch = 1 ; sch <= N ; sch *= 9) {
        OUT[0] = 0;
        TESTD2("omp target teams distribute parallel for simd num_teams(tms) thread_limit(ths) dist_schedule(static,sch) schedule(static,sch) REDUCTION_MAP REDUCTION_CLAUSES",
          {
            REDUCTION_INIT();
          },
          REDUCTION_LOOP(),
          {
            REDUCTION_FINAL();
          },
          VERIFY(0, 1, OUT[i], (trial+1) * EXPECTED[i]));
      }
    }
  }

  //
  // Test: reduction on target teams distribute parallel for simd with dist_schedule static chunk, schedule dynamic nochunk.
  //
  for (int tms = 1 ; tms <= 512 ; tms *= 7) {
    for (int ths = 1 ; ths <= 1024 ; ths *= 9) {
      for(int sch = 1 ; sch <= N ; sch *= 9) {
        OUT[0] = 0;
        TESTD2("omp target teams distribute parallel for simd num_teams(tms) thread_limit(ths) dist_schedule(static,sch) schedule(dynamic) REDUCTION_MAP REDUCTION_CLAUSES",
          {
            REDUCTION_INIT();
          },
          REDUCTION_LOOP(),
          {
            REDUCTION_FINAL();
          },
          VERIFY(0, 1, OUT[i], (trial+1) * EXPECTED[i]));
      }
    }
  }

  //
  // Test: reduction on target teams distribute parallel for simd with dist_schedule static chunk,schedule dynamic chunk.
  //
  for (int tms = 1 ; tms <= 512 ; tms *= 7) {
    for (int ths = 1 ; ths <= 1024 ; ths *= 9) {
      for(int sch = 1 ; sch <= N ; sch *= 9) {
        OUT[0] = 0;
        TESTD2("omp target teams distribute parallel for simd num_teams(tms) thread_limit(ths) dist_schedule(static,sch) schedule(dynamic,sch) REDUCTION_MAP REDUCTION_CLAUSES",
          {
            REDUCTION_INIT();
          },
          REDUCTION_LOOP(),
          {
            REDUCTION_FINAL();
          },
          VERIFY(0, 1, OUT[i], (trial+1) * EXPECTED[i]));
      }
    }
  }

  //
  // Test: reduction on target teams distribute parallel for simd with dist_schedule static chunk, schedule guided nochunk.
  //
  for (int tms = 1 ; tms <= 512 ; tms *= 7) {
    for (int ths = 1 ; ths <= 1024 ; ths *= 9) {
      for(int sch = 1 ; sch <= N ; sch *= 9) {
        OUT[0] = 0;
        TESTD2("omp target teams distribute parallel for simd num_teams(tms) thread_limit(ths) dist_schedule(static,sch) schedule(guided) REDUCTION_MAP REDUCTION_CLAUSES",
          {
            REDUCTION_INIT();
          },
          REDUCTION_LOOP(),
          {
            REDUCTION_FINAL();
          },
          VERIFY(0, 1, OUT[i], (trial+1) * EXPECTED[i]));
      }
    }
  }

  //
  // Test: reduction on target teams distribute parallel for simd with dist_schedule static chunk, schedule guided chunk.
  //
  for (int tms = 1 ; tms <= 512 ; tms *= 7) {
    for (int ths = 1 ; ths <= 1024 ; ths *= 9) {
      for(int sch = 1 ; sch <= N ; sch *= 9) {
        OUT[0] = 0;
        TESTD2("omp target teams distribute parallel for simd num_teams(tms) thread_limit(ths) dist_schedule(static,sch) schedule(guided,sch) REDUCTION_MAP REDUCTION_CLAUSES",
          {
            REDUCTION_INIT();
          },
          REDUCTION_LOOP(),
          {
            REDUCTION_FINAL();
          },
          VERIFY(0, 1, OUT[i], (trial+1) * EXPECTED[i]));
      }
    }
  }

  //
  // Test: reduction on target teams distribute parallel for simd with dist_schedule static chunk, schedule auto.
  //
  for (int tms = 1 ; tms <= 512 ; tms *= 7) {
    for (int ths = 1 ; ths <= 1024 ; ths *= 9) {
      for(int sch = 1 ; sch <= N ; sch *= 9) {
        OUT[0] = 0;
        TESTD2("omp target teams distribute parallel for simd num_teams(tms) thread_limit(ths) dist_schedule(static,sch) schedule(dynamic) REDUCTION_MAP REDUCTION_CLAUSES",
          {
            REDUCTION_INIT();
          },
          REDUCTION_LOOP(),
          {
            REDUCTION_FINAL();
          },
          VERIFY(0, 1, OUT[i], (trial+1) * EXPECTED[i]));
      }
    }
  }

  //
  // Test: reduction on target teams distribute parallel for simd with dist_schedule static chunk, schedule runtime.
  //
  for (int tms = 1 ; tms <= 512 ; tms *= 7) {
    for (int ths = 1 ; ths <= 1024 ; ths *= 9) {
      for(int sch = 1 ; sch <= N ; sch *= 9) {
        OUT[0] = 0;
        TESTD2("omp target teams distribute parallel for simd num_teams(tms) thread_limit(ths) dist_schedule(static,sch) schedule(runtime) REDUCTION_MAP REDUCTION_CLAUSES",
          {
            REDUCTION_INIT();
          },
          REDUCTION_LOOP(),
          {
            REDUCTION_FINAL();
          },
          VERIFY(0, 1, OUT[i], (trial+1) * EXPECTED[i]));
      }
    }
  }

  //
  // Test: reduction on teams distribute parallel for simd.
  //
  for (int tms = 1 ; tms <= 512 ; tms *= 7) {
    for (int ths = 1 ; ths <= 1024 ; ths *= 9) {
      OUT[0] = 0;
      TESTD2("omp target REDUCTION_MAP",
        {
          REDUCTION_INIT();
        },
        {
        _Pragma("teams distribute parallel for simd num_teams(tms) thread_limit(ths) REDUCTION_CLAUSES")
        REDUCTION_LOOP()
        },
        {
          REDUCTION_FINAL();
        },
        VERIFY(0, 1, OUT[i], (trial+1) * EXPECTED[i]));
    }
  }

  //
  // Test: reduction on teams distribute parallel for simd with schedule static nochunk.
  //
  for (int tms = 1 ; tms <= 512 ; tms *= 7) {
    for (int ths = 1 ; ths <= 1024 ; ths *= 9) {
      OUT[0] = 0;
      TESTD2("omp target REDUCTION_MAP",
        {
          REDUCTION_INIT();
        },
        {
        _Pragma("teams distribute parallel for simd num_teams(tms) thread_limit(ths) schedule(static) REDUCTION_CLAUSES")
        REDUCTION_LOOP()
        },
        {
          REDUCTION_FINAL();
        },
        VERIFY(0, 1, OUT[i], (trial+1) * EXPECTED[i]));
    }
  }

  //
  // Test: reduction on teams distribute parallel for simd with schedule static chunk.
  //
  for (int tms = 1 ; tms <= 512 ; tms *= 7) {
    for (int ths = 1 ; ths <= 1024 ; ths *= 9) {
      for(int sch = 1 ; sch <= N ; sch *= 9) {
        OUT[0] = 0;
        TESTD2("omp target REDUCTION_MAP",
          {
            REDUCTION_INIT();
          },
          {
          _Pragma("teams distribute parallel for simd num_teams(tms) thread_limit(ths) schedule(static,sch) REDUCTION_CLAUSES")
          REDUCTION_LOOP()
          },
          {
            REDUCTION_FINAL();
          },
          VERIFY(0, 1, OUT[i], (trial+1) * EXPECTED[i]));
      }
    }
  }

  //
  // Test: reduction on teams distribute parallel for simd with schedule dynamic nochunk.
  //
  for (int tms = 1 ; tms <= 512 ; tms *= 7) {
    for (int ths = 1 ; ths <= 1024 ; ths *= 9) {
      OUT[0] = 0;
      TESTD2("omp target REDUCTION_MAP",
        {
          REDUCTION_INIT();
        },
        {
        _Pragma("teams distribute parallel for simd num_teams(tms) thread_limit(ths) schedule(dynamic) REDUCTION_CLAUSES")
        REDUCTION_LOOP()
        },
        {
          REDUCTION_FINAL();
        },
        VERIFY(0, 1, OUT[i], (trial+1) * EXPECTED[i]));
    }
  }

  //
  // Test: reduction on teams distribute parallel for simd with schedule dynamic chunk.
  //
  for (int tms = 1 ; tms <= 512 ; tms *= 7) {
    for (int ths = 1 ; ths <= 1024 ; ths *= 9) {
      for(int sch = 1 ; sch <= N ; sch *= 9) {
        OUT[0] = 0;
        TESTD2("omp target REDUCTION_MAP",
          {
            REDUCTION_INIT();
          },
          {
          _Pragma("teams distribute parallel for simd num_teams(tms) thread_limit(ths) schedule(dynamic,sch) REDUCTION_CLAUSES")
          REDUCTION_LOOP()
          },
          {
            REDUCTION_FINAL();
          },
          VERIFY(0, 1, OUT[i], (trial+1) * EXPECTED[i]));
      }
    }
  }

  //
  // Test: reduction on teams distribute parallel for simd with dist_schedule static nochunk.
  //
  for (int tms = 1 ; tms <= 512 ; tms *= 7) {
    for (int ths = 1 ; ths <= 1024 ; ths *= 9) {
      OUT[0] = 0;
      TESTD2("omp target REDUCTION_MAP",
        {
          REDUCTION_INIT();
        },
        {
        _Pragma("teams distribute parallel for simd num_teams(tms) thread_limit(ths) dist_schedule(static) REDUCTION_CLAUSES")
        REDUCTION_LOOP()
        },
        {
          REDUCTION_FINAL();
        },
        VERIFY(0, 1, OUT[i], (trial+1) * EXPECTED[i]));
    }
  }

  //
  // Test: reduction on teams distribute parallel for simd with dist_schedule static nochunk, schedule static nochunk.
  //
  for (int tms = 1 ; tms <= 512 ; tms *= 7) {
    for (int ths = 1 ; ths <= 1024 ; ths *= 9) {
      OUT[0] = 0;
      TESTD2("omp target REDUCTION_MAP",
        {
          REDUCTION_INIT();
        },
        {
        _Pragma("teams distribute parallel for simd num_teams(tms) thread_limit(ths) dist_schedule(static) schedule(static) REDUCTION_CLAUSES")
        REDUCTION_LOOP()
        },
        {
          REDUCTION_FINAL();
        },
        VERIFY(0, 1, OUT[i], (trial+1) * EXPECTED[i]));
    }
  }

  //
  // Test: reduction on teams distribute parallel for simd with dist_schedule static nochunk, schedule static chunk.
  //
  for (int tms = 1 ; tms <= 512 ; tms *= 7) {
    for (int ths = 1 ; ths <= 1024 ; ths *= 9) {
      for(int sch = 1 ; sch <= N ; sch *= 9) {
        OUT[0] = 0;
        TESTD2("omp target REDUCTION_MAP",
          {
            REDUCTION_INIT();
          },
          {
          _Pragma("teams distribute parallel for simd num_teams(tms) thread_limit(ths) dist_schedule(static) schedule(static,sch) REDUCTION_CLAUSES")
          REDUCTION_LOOP()
          },
          {
            REDUCTION_FINAL();
          },
          VERIFY(0, 1, OUT[i], (trial+1) * EXPECTED[i]));
      }
    }
  }

  //
  // Test: reduction on teams distribute parallel for simd with dist_schedule static chunk, schedule static nochunk.
  //
  for (int tms = 1 ; tms <= 512 ; tms *= 7) {
    for (int ths = 1 ; ths <= 1024 ; ths *= 9) {
      for(int sch = 1 ; sch <= N ; sch *= 9) {
        OUT[0] = 0;
        TESTD2("omp target REDUCTION_MAP",
          {
            REDUCTION_INIT();
          },
          {
          _Pragma("teams distribute parallel for simd num_teams(tms) thread_limit(ths) dist_schedule(static,sch) schedule(static) REDUCTION_CLAUSES")
          REDUCTION_LOOP()
          },
          {
            REDUCTION_FINAL();
          },
          VERIFY(0, 1, OUT[i], (trial+1) * EXPECTED[i]));
      }
    }
  }

  //
  // Test: reduction on teams distribute parallel for simd with dist_schedule static chunk, schedule static chunk.
  //
  for (int tms = 1 ; tms <= 512 ; tms *= 7) {
    for (int ths = 1 ; ths <= 1024 ; ths *= 9) {
      for(int sch = 1 ; sch <= N ; sch *= 9) {
        OUT[0] = 0;
        TESTD2("omp target REDUCTION_MAP",
          {
            REDUCTION_INIT();
          },
          {
          _Pragma("teams distribute parallel for simd num_teams(tms) thread_limit(ths) dist_schedule(static,sch) schedule(static,sch) REDUCTION_CLAUSES")
          REDUCTION_LOOP()
          },
          {
            REDUCTION_FINAL();
          },
          VERIFY(0, 1, OUT[i], (trial+1) * EXPECTED[i]));
      }
    }
  }

  //
  // Test: reduction on teams distribute parallel for simd with dist_schedule static chunk, schedule dynamic nochunk.
  //
  for (int tms = 1 ; tms <= 512 ; tms *= 7) {
    for (int ths = 1 ; ths <= 1024 ; ths *= 9) {
      for(int sch = 1 ; sch <= N ; sch *= 9) {
        OUT[0] = 0;
        TESTD2("omp target REDUCTION_MAP",
          {
            REDUCTION_INIT();
          },
          {
          _Pragma("teams distribute parallel for simd num_teams(tms) thread_limit(ths) dist_schedule(static,sch) schedule(dynamic) REDUCTION_CLAUSES")
          REDUCTION_LOOP()
          },
          {
            REDUCTION_FINAL();
          },
          VERIFY(0, 1, OUT[i], (trial+1) * EXPECTED[i]));
      }
    }
  }

  //
  // Test: reduction on teams distribute parallel for simd with dist_schedule static chunk,schedule dynamic chunk.
  //
  for (int tms = 1 ; tms <= 512 ; tms *= 7) {
    for (int ths = 1 ; ths <= 1024 ; ths *= 9) {
      for(int sch = 1 ; sch <= N ; sch *= 9) {
        OUT[0] = 0;
        TESTD2("omp target REDUCTION_MAP",
          {
            REDUCTION_INIT();
          },
          {
          _Pragma("teams distribute parallel for simd num_teams(tms) thread_limit(ths) dist_schedule(static,sch) schedule(dynamic,sch) REDUCTION_CLAUSES")
          REDUCTION_LOOP()
          },
          {
            REDUCTION_FINAL();
          },
          VERIFY(0, 1, OUT[i], (trial+1) * EXPECTED[i]));
      }
    }
  }

  //
  // Test: reduction on teams distribute parallel for simd with dist_schedule static chunk, schedule guided nochunk.
  //
  for (int tms = 1 ; tms <= 512 ; tms *= 7) {
    for (int ths = 1 ; ths <= 1024 ; ths *= 9) {
      for(int sch = 1 ; sch <= N ; sch *= 9) {
        OUT[0] = 0;
        TESTD2("omp target REDUCTION_MAP",
          {
            REDUCTION_INIT();
          },
          {
          _Pragma("teams distribute parallel for simd num_teams(tms) thread_limit(ths) dist_schedule(static,sch) schedule(guided) REDUCTION_CLAUSES")
          REDUCTION_LOOP()
          },
          {
            REDUCTION_FINAL();
          },
          VERIFY(0, 1, OUT[i], (trial+1) * EXPECTED[i]));
      }
    }
  }

  //
  // Test: reduction on teams distribute parallel for simd with dist_schedule static chunk, schedule guided chunk.
  //
  for (int tms = 1 ; tms <= 512 ; tms *= 7) {
    for (int ths = 1 ; ths <= 1024 ; ths *= 9) {
      for(int sch = 1 ; sch <= N ; sch *= 9) {
        OUT[0] = 0;
        TESTD2("omp target REDUCTION_MAP",
          {
            REDUCTION_INIT();
          },
          {
          _Pragma("teams distribute parallel for simd num_teams(tms) thread_limit(ths) dist_schedule(static,sch) schedule(guided,sch) REDUCTION_CLAUSES")
          REDUCTION_LOOP()
          },
          {
            REDUCTION_FINAL();
          },
          VERIFY(0, 1, OUT[i], (trial+1) * EXPECTED[i]));
      }
    }
  }

  //
  // Test: reduction on teams distribute parallel for simd with dist_schedule static chunk, schedule auto.
  //
  for (int tms = 1 ; tms <= 512 ; tms *= 7) {
    for (int ths = 1 ; ths <= 1024 ; ths *= 9) {
      for(int sch = 1 ; sch <= N ; sch *= 9) {
        OUT[0] = 0;
        TESTD2("omp target REDUCTION_MAP",
          {
            REDUCTION_INIT();
          },
          {
          _Pragma("teams distribute parallel for simd num_teams(tms) thread_limit(ths) dist_schedule(static,sch) schedule(dynamic) REDUCTION_CLAUSES")
          REDUCTION_LOOP()
          },
          {
            REDUCTION_FINAL();
          },
          VERIFY(0, 1, OUT[i], (trial+1) * EXPECTED[i]));
      }
    }
  }

  //
  // Test: reduction on teams distribute parallel for simd with dist_schedule static chunk, schedule runtime.
  //
  for (int tms = 1 ; tms <= 512 ; tms *= 7) {
    for (int ths = 1 ; ths <= 1024 ; ths *= 9) {
      for(int sch = 1 ; sch <= N ; sch *= 9) {
        OUT[0] = 0;
        TESTD2("omp target REDUCTION_MAP",
          {
            REDUCTION_INIT();
          },
          {
          _Pragma("teams distribute parallel for simd num_teams(tms) thread_limit(ths) dist_schedule(static,sch) schedule(runtime) REDUCTION_CLAUSES")
          REDUCTION_LOOP()
          },
          {
            REDUCTION_FINAL();
          },
          VERIFY(0, 1, OUT[i], (trial+1) * EXPECTED[i]));
      }
    }
  }

  return 0;
}

#ifndef __UTILITIES_H__
#define __UTILITIES_H__

#define ZERO_ARRAY(UB, X) { \
  int i; \
  for (i = 0; i < UB; i++) { \
    X[i] = 0; \
  } \
}

#define INIT_LOOP(UB, B) { \
  int i; \
  for (i = 0; i < UB; i++) { \
    B \
  } \
}

#define VERIFY_E(LB, UB, X, VAL, EPSILON) { \
  int i; \
  int errorNum = 0 ;     \
  for (i = LB; i < UB; i++) { \
    if (fabs(X - VAL) > EPSILON) { \
      printf ("Failed at %3d with %s, expected %f, got %f\n", i, #X, VAL, X); \
      fail = 1; \
      if (errorNum++>10) break;   \
    } \
  } \
}

#define VERIFY(LB, UB, X, VAL) { \
  int i; \
  int errorNum = 0 ;     \
  for (i = LB; i < UB; i++) { \
    if (X != VAL) { \
      printf ("Failed at %3d with %s, expected %f, got %f\n", i, #X, VAL, X); \
      fail = 1; \
      if (errorNum++>10) break;   \
    } \
  } \
}

#define VERIFY_ARRAY(LB, UB, X, Y) { \
  int i; \
  int errorNum = 0 ;     \
  for (i = LB; i < UB; i++) { \
    if (X[i] != Y[i]) { \
      printf ("Failed at %3d, expected %d, got %d\n", i, X[i], Y[i]); \
      fail = 1; \
      if (errorNum++>10) break;   \
    } \
  } \
}

#define TEST(T, V) { \
  int fail = 0; \
  int trial; \
  for (int trial = 0; trial < TRIALS && fail == 0; trial++) { \
    _Pragma("omp target teams num_teams(1) thread_limit(1024)") \
     {T} \
    V \
  } \
  if (fail) { \
    printf ("Failed\n"); \
  } else { \
    printf ("Succeeded\n"); \
  } \
}

#define DUMP_SUCCESS(N) { \
  for (int i = 0; i < (N); i++) \
    printf ("Succeeded\n"); \
}

#define TESTD(D, T, V) { \
  int fail = 0; \
  int trial; \
  for (int trial = 0; trial < TRIALS && fail == 0; trial++) { \
    _Pragma(D) \
     {T} \
    V \
  } \
  if (fail) { \
    printf ("Failed\n"); \
  } else { \
    printf ("Succeeded\n"); \
  } \
}

#define TESTD2(D, PRE, T, POST, V) { \
  int fail = 0; \
  int trial; \
  for (int trial = 0; trial < TRIALS && fail == 0; trial++) { \
    PRE \
    _Pragma(D) \
     {T} \
    POST \
    V \
  } \
  if (fail) { \
    printf ("Failed\n"); \
  } else { \
    printf ("Succeeded\n"); \
  } \
}

#define TEST_MAP(_I, _P, _T, _V) {                   \
  int fail = 0; \
  int trial; \
  for (int trial = 0; trial < TRIALS && fail == 0; trial++) { \
    { _I }  \
    _P \
     { _T } \
    { _V }  \
  } \
  if (fail) { \
    printf ("Failed\n"); \
  } else { \
    printf ("Succeeded\n"); \
  } \
}

#endif

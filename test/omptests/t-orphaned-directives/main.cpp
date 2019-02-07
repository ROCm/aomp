#include <stdlib.h>
#include <stdio.h>

#include <omp.h>

#include "../utilities/check.h"
#include "../utilities/utilities.h"

#define MAX_N 25000

double *a, *a_h, *b, *c;

void reset_input(double *a, double *a_h, double *b, double *c) {
  for(int i = 0 ; i < MAX_N ; i++) {
    a[i] = a_h[i] = i;
    b[i] = i*2;
    c[i] = i-3;
  }
}

#pragma omp declare target
void kernel1(int num_threads, double *RESULT, int *VALID,
             int offset, int factor, int N,
             double *Ad, double *Bd, double *Cd,
             long long *OUT, int *num_tests);

void add_f1(double *a, double *b, double *c, int n, int sch);
void add_f2(double *a, double *b, double *c, int n, int sch);
void add_f3(double *a, double *b, double *c, int n, int sch);
void add_f4(double *a, double *b, double *c, int n, int sch);

void add_dpf1(double *a, double *b, double *c, int n, int sch);
void add_dpf2(double *a, double *b, double *c, int n, int sch);
void add_dpf3(double *a, double *b, double *c, int n, int sch);
void add_dpf4(double *a, double *b, double *c, int n, int sch);

template <typename T>
void tadd_dpf1(T *a, T *b, T *c, int n, int sch) {
#pragma omp distribute parallel for dist_schedule(static,sch) schedule(static)
  for (int i = 0; i < n; ++i) {
    a[i] += b[i] + c[i];
  }
}
#pragma omp end declare target

int main() {
  check_offloading();

  int cpuExec = 0;
  #pragma omp target map(tofrom: cpuExec)
  {
    cpuExec = omp_is_initial_device();
  }
  int max_teams = 256;
  int gpu_threads = 256;
  int cpu_threads = 32;
  int max_threads = cpuExec ? cpu_threads : gpu_threads;

  a = (double *) malloc(MAX_N * sizeof(double));
  a_h = (double *) malloc(MAX_N * sizeof(double));
  b = (double *) malloc(MAX_N * sizeof(double));
  c = (double *) malloc(MAX_N * sizeof(double));

#pragma omp target enter data map(to:a[:MAX_N],b[:MAX_N],c[:MAX_N])

  for (int n = 32 ; n < MAX_N ; n+=5000) {
    int t = 0;
    reset_input(a, a_h, b, c);

#pragma omp target update to(a[:n],b[:n],c[:n])

    for (int ths = 1; ths <= 1024; ths *= 3) {
      for(int sch = 1 ; sch <= n ; sch *= 1200) {
        t+=4;
#pragma omp target
#pragma omp parallel
        {
          add_f1(a, b, c, n, sch);
          add_f2(a, b, c, n, sch);
          add_f3(a, b, c, n, sch);
          add_f4(a, b, c, n, sch);
        }
      }
    }

    // check results for each 'n'
    for (int times = 0 ; times < t ; times++)
      for (int i = 0; i < n; ++i)
        a_h[i] += b[i] + c[i];

#pragma omp target update from(a[:n])

    for (int i = 0; i < n; ++i) {
      if (a_h[i] != a[i]) {
        printf("Error at n = %d, i = %d: host = %lf, device = %lf\n", n, i, a_h[i], a[i]);
        return 1;
      }
    }
  } // loop 'n'
  printf("Succeeded\n");


  for (int n = 32 ; n < MAX_N ; n+=5000) {
    int t = 0;
    reset_input(a, a_h, b, c);

#pragma omp target update to(a[:n],b[:n],c[:n])

    for (int ths = 1; ths <= 1024; ths *= 3) {
      for(int sch = 1 ; sch <= n ; sch *= 1200) {
        t+=4;
#pragma omp target parallel num_threads(1024)
        {
          add_f1(a, b, c, n, sch);
          add_f2(a, b, c, n, sch);
          add_f3(a, b, c, n, sch);
          add_f4(a, b, c, n, sch);
        }
      }
    }

    // check results for each 'n'
    for (int times = 0 ; times < t ; times++)
      for (int i = 0; i < n; ++i)
        a_h[i] += b[i] + c[i];

#pragma omp target update from(a[:n])

    for (int i = 0; i < n; ++i) {
      if (a_h[i] != a[i]) {
        printf("Error at n = %d, i = %d: host = %lf, device = %lf\n", n, i, a_h[i], a[i]);
        return 1;
      }
    }
  } // loop 'n'
  printf("Succeeded\n");


  for (int n = 32 ; n < MAX_N ; n+=5000) {
    int t = 0;
    reset_input(a, a_h, b, c);

#pragma omp target update to(a[:n],b[:n],c[:n])

    for (int tms = 1 ; tms <= 256 ; tms *= 2) { // 8 times
      for (int ths = 32 ; ths <= 1024 ; ths *= 2) { // 6 times
        for(int sch = 1 ; sch <= n ; sch *= 1200) {
          t+=4;
#pragma omp target teams num_teams(tms) thread_limit(ths)
          {
            tadd_dpf1<double>(a, b, c, n, sch);
            add_dpf2(a, b, c, n, sch);
            add_dpf3(a, b, c, n, sch);
            add_dpf4(a, b, c, n, sch);
          }
        } // loop 'sch'
      } // loop 'ths'
    } // loop 'tms'

    // check results for each 'n'
    for (int times = 0 ; times < t ; times++)
      for (int i = 0; i < n; ++i)
        a_h[i] += b[i] + c[i];

#pragma omp target update from(a[:n])

    for (int i = 0; i < n; ++i) {
      if (a_h[i] != a[i]) {
        printf("Error at n = %d, i = %d: host = %lf, device = %lf\n", n, i, a_h[i], a[i]);
        return 1;
      }
    }
  } // loop 'n'
  printf("Succeeded\n");

#pragma omp target exit data map(release:a[:MAX_N],b[:MAX_N],c[:MAX_N])



  #define N (957*3)
  double Ad[N], Bd[N], Cd[N];

  #define INIT() { \
    INIT_LOOP(N, { \
      Ad[i] = 1 << 16; \
      Bd[i] = i << 16; \
      Cd[i] = -(i << 16); \
    }) \
  }

  INIT();

  double RESULT[256];
  int VALID[256];
  long long EXPECTED[7];
  EXPECTED[0] = 34; EXPECTED[1] = 2311; EXPECTED[2] = 4795;
  EXPECTED[3] = 7532; EXPECTED[4] = 10468; EXPECTED[5] = 12999;
  EXPECTED[6] = 15345;
  unsigned e = 0;
  for (int t = 2; t <= max_threads; t+=39) {
    long long OUT = 0;
    int num_threads = t;
    int num_tests = 0;
    #pragma omp target teams map(tofrom: OUT, num_tests) num_teams(1) thread_limit(max_threads)
    {
      #pragma omp parallel num_threads(num_threads)
      {
        for (int offset = 0; offset < 32; offset++) {
          for (int factor = 1; factor < 33; factor++) {
            kernel1(num_threads, RESULT, VALID, offset, factor,
                    N, Ad, Bd, Cd, &OUT, &num_tests);
          }
        }
      }
    }

    if (OUT + num_tests != EXPECTED[e++])
      printf ("Failed test with num_threads = %d, OUT + num_tests = %ld\n",
              t, OUT + num_tests);
    else
      printf ("Succeeded\n");
  }
  if (cpuExec) {
    DUMP_SUCCESS(6);
  }


  e = 0;
  for (int t = 2; t <= max_threads; t+=39) {
    long long OUT = 0;
    int num_threads = t;
    int num_tests = 0;
    #pragma omp target parallel map(tofrom: OUT, num_tests) num_threads(num_threads)
    {
        for (int offset = 0; offset < 32; offset++) {
          for (int factor = 1; factor < 33; factor++) {
            kernel1(num_threads, RESULT, VALID, offset, factor,
                    N, Ad, Bd, Cd, &OUT, &num_tests);
          }
        }
    }

    if (OUT + num_tests != EXPECTED[e++])
      printf ("Failed test with num_threads = %d, OUT + num_tests = %ld\n",
              t, OUT + num_tests);
    else
      printf ("Succeeded\n");
  }
  if (cpuExec) {
    DUMP_SUCCESS(6);
  }


  long long OUT = 0;
  int num_tests = 0;
  #pragma omp target map(tofrom: OUT, num_tests)
  {
    kernel1(1, RESULT, VALID, 0, 1,
            N, Ad, Bd, Cd, &OUT, &num_tests);
  }

  if (OUT + num_tests != 1)
    printf ("Failed test with OUT + num_tests = %ld\n",
            OUT + num_tests);
  else
    printf ("Succeeded\n");

  return 0;
}

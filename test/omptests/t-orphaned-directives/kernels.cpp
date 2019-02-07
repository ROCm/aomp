
#include <stdlib.h>
#include <stdio.h>
#include <omp.h>

#pragma omp declare target
void kernel1(int num_threads, double *RESULT, int *VALID,
             int offset, int factor, int N,
             double *Ad, double *Bd, double *Cd,
             long long *OUT, int *num_tests) {
  double Rd1 = 0; double Rd2 = 0;
  int tid = omp_get_thread_num();
  int lid = tid % 32;
  VALID[tid] = 0;

  if (lid >= offset && lid % factor == 0) {
    #pragma omp parallel for reduction(+:Rd1) reduction(-:Rd2)
    for (int i = 0; i < N; i++) {
      Rd1 += Ad[i] + (Bd[i] + Cd[i]);
      Rd2 += Ad[i] + (Bd[i] + Cd[i]);
    }
    VALID[tid] = 1;
    RESULT[tid] = Rd1 + Rd2;
  }
  #pragma omp barrier
  if (tid == 0) {
    for (int i = 0; i < num_threads; i++) {
      if (VALID[i]) (*num_tests)++;
      if (VALID[i] && (RESULT[i] - (double) ((2*N) << 16) > .001)) {
        *OUT = 1;
        printf ("Failed nested parallel reduction\n");
      }
    }
  }
  #pragma omp barrier
}
#pragma omp end declare target

#pragma omp declare target
void add_f1(double *a, double *b, double *c, int n, int sch) {
#pragma omp for nowait
  for (int i = 0; i < n; ++i) {
    a[i] += b[i] + c[i];
  }
}

void add_f2(double *a, double *b, double *c, int n, int sch) {
#pragma omp for nowait
  for (int i = 0; i < n; ++i) {
    a[i] += b[i] + c[i];
  }
}

void add_f3(double *a, double *b, double *c, int n, int sch) {
#pragma omp barrier
#pragma omp for schedule(guided)
  for (int i = 0; i < n; ++i) {
    a[i] += b[i] + c[i];
  }
}

void add_f4(double *a, double *b, double *c, int n, int sch) {
#pragma omp for schedule(dynamic, sch)
  for (int i = 0; i < n; ++i) {
    a[i] += b[i] + c[i];
  }
}

void add_dpf1(double *a, double *b, double *c, int n, int sch) {
#pragma omp distribute parallel for dist_schedule(static,sch) schedule(static)
  for (int i = 0; i < n; ++i) {
    a[i] += b[i] + c[i];
  }
}

void add_dpf2(double *a, double *b, double *c, int n, int sch) {
#pragma omp distribute parallel for dist_schedule(static,sch) schedule(static, sch)
  for (int i = 0; i < n; ++i) {
    a[i] += b[i] + c[i];
  }
}

void add_dpf3(double *a, double *b, double *c, int n, int sch) {
#pragma omp distribute parallel for dist_schedule(static,sch) schedule(runtime)
  for (int i = 0; i < n; ++i) {
    a[i] += b[i] + c[i];
  }
}

void add_dpf4(double *a, double *b, double *c, int n, int sch) {
#pragma omp distribute parallel for dist_schedule(static,sch) schedule(dynamic, sch)
  for (int i = 0; i < n; ++i) {
    a[i] += b[i] + c[i];
  }
}
#pragma omp end declare target

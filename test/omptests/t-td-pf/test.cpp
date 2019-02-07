#include <omp.h>
#include <stdio.h>

#define MAX_N 25000
#define SIZE 1000

void reset_input(double *a, double *a_h, double *b, double *c) {
  for(int i = 0 ; i < MAX_N ; i++) {
    a[i] = a_h[i] = i;
    b[i] = i*2;
    c[i] = i-3;
  }
}

int check_results(int t, int n, double* a_h, double* a, double* b, double* c){
  // check results for each 'n'
  for (int times = 0 ; times < t ; times++)
    for (int i = 0; i < n; ++i)
      a_h[i] += (b[i] + c[i]);
  #pragma omp target update from(a[:n])
  for (int i = 0; i < n; ++i) {
    if (a_h[i] != a[i]) {
      printf("Error at n = %d, i = %d: host = %lf, device = %lf\n", n, i, a_h[i], a[i]);
      return 0;
    }
  }
  return 1;
}

#define CODE(dist_step, step) \
  success = 0; \
  for (int n = 32 ; n < MAX_N ; n+=5000) { \
    reset_input(a, a_h, b, c); \
    _Pragma("omp target update to(a[:n],b[:n],c[:n])") \
    t = 0; \
    for (int tms = 1 ; tms <= 256 ; tms *= 2) { \
      for (int ths = 32 ; ths <= 1024 ; ths *= 2) { \
        for(int dssch = 1000 ; dssch <= n ; dssch *= dist_step) { \
          for(int sch = 1 ; sch <= n ; sch *= step) { \
            t++; \
            _Pragma("omp target") \
            _Pragma("omp teams distribute num_teams(tms) thread_limit(ths) DIST_SCHEDULE") \
            for (int i=0; i<n; i+=blockSize) { \
              int ub = (i+blockSize > n) ? n : i+blockSize; \
              _Pragma("omp parallel for SCHEDULE") \
              for (int j=i ; j < ub; j++) { \
                a[j] += b[j] + c[j]; \
              } \
            } \
          } \
        } \
      } \
    } \
    success += check_results(t, n, a_h, a, b, c); \
  } \
  if (success == expected) \
    printf("Succeeded\n");

int main(int argc, char *argv[]) {
  double * a = (double *) malloc(MAX_N * sizeof(double));
  double * a_h = (double *) malloc(MAX_N * sizeof(double));
  double * b = (double *) malloc(MAX_N * sizeof(double));
  double * c = (double *) malloc(MAX_N * sizeof(double));
  int success = 0;
  int blockSize = 32;
  int expected = MAX_N / 5000;
  int t = 0;

  #pragma omp target enter data map(to:a[:MAX_N],b[:MAX_N],c[:MAX_N])

  printf("-, no schedule clauses\n");
  #define DIST_SCHEDULE  
  #define SCHEDULE 
  CODE(3000, 1200)
  #undef DIST_SCHEDULE
  #undef SCHEDULE

  printf("-, schedule static no chunk\n");
  #define DIST_SCHEDULE  
  #define SCHEDULE schedule(static)
  CODE(3000, 1200)
  #undef DIST_SCHEDULE
  #undef SCHEDULE

  printf("-, schedule dynamic no chunk\n");
  #define DIST_SCHEDULE  
  #define SCHEDULE schedule(dynamic)
  CODE(3000, 1200)
  #undef DIST_SCHEDULE
  #undef SCHEDULE

  printf("-, schedule static chunk\n");
  #define DIST_SCHEDULE  
  #define SCHEDULE schedule(static, sch)
  CODE(3000, 3000)
  #undef DIST_SCHEDULE
  #undef SCHEDULE

  printf("-, schedule dynamic chunk\n");
  #define DIST_SCHEDULE  
  #define SCHEDULE schedule(dynamic, sch)
  CODE(3000, 1200)
  #undef DIST_SCHEDULE
  #undef SCHEDULE

  printf("dist_schedule(static), -\n");
  #define DIST_SCHEDULE dist_schedule(static)
  #define SCHEDULE
  CODE(3000, 3000)
  #undef DIST_SCHEDULE
  #undef SCHEDULE

  printf("dist_schedule(static, sch), -\n");
  #define DIST_SCHEDULE dist_schedule(static, sch)
  #define SCHEDULE
  CODE(10000, 10000)
  #undef DIST_SCHEDULE
  #undef SCHEDULE

  printf("dist_schedule(static), schedule(static)\n");
  #define DIST_SCHEDULE dist_schedule(static)
  #define SCHEDULE schedule(static)
  CODE(30000, 30000)
  #undef DIST_SCHEDULE
  #undef SCHEDULE

  printf("dist_schedule(static), schedule(static, sch)\n");
  #define DIST_SCHEDULE dist_schedule(static)
  #define SCHEDULE schedule(static, sch)
  CODE(30000, 1000)
  #undef DIST_SCHEDULE
  #undef SCHEDULE

  printf("dist_schedule(static), schedule(static, sch)\n");
  #define DIST_SCHEDULE dist_schedule(static, sch)
  #define SCHEDULE schedule(static)
  CODE(30000, 1000)
  #undef DIST_SCHEDULE
  #undef SCHEDULE

  printf("dist_schedule(static, dssch), schedule(static, sch)\n");
  #define DIST_SCHEDULE dist_schedule(static, dssch)
  #define SCHEDULE schedule(static, sch)
  CODE(1200, 3000)
  #undef DIST_SCHEDULE
  #undef SCHEDULE

  printf("dist_schedule(static, sch), schedule(dynamic)\n");
  #define DIST_SCHEDULE dist_schedule(static, sch)
  #define SCHEDULE schedule(dynamic)
  CODE(30000, 3000)
  #undef DIST_SCHEDULE
  #undef SCHEDULE

  printf("dist_schedule(static, dssch), schedule(dynamic, sch)\n");
  #define DIST_SCHEDULE dist_schedule(static, dssch)
  #define SCHEDULE schedule(dynamic, sch)
  CODE(3000, 3000)
  #undef DIST_SCHEDULE
  #undef SCHEDULE

  printf("dist_schedule(static, sch), schedule guided\n");
  #define DIST_SCHEDULE dist_schedule(static, sch)
  #define SCHEDULE schedule(guided)
  CODE(30000, 3000)
  #undef DIST_SCHEDULE
  #undef SCHEDULE

  printf("dist_schedule(static, dssch), schedule(guided, sch)\n");
  #define DIST_SCHEDULE dist_schedule(static, dssch)
  #define SCHEDULE schedule(guided, sch)
  CODE(3000, 3000)
  #undef DIST_SCHEDULE
  #undef SCHEDULE

  printf("dist_schedule(static, sch), schedule runtime\n");
  #define DIST_SCHEDULE dist_schedule(static, sch)
  #define SCHEDULE schedule(runtime)
  CODE(3000, 1200)
  #undef DIST_SCHEDULE
  #undef SCHEDULE

  printf("dist_schedule(static, sch), schedule auto\n");
  #define DIST_SCHEDULE dist_schedule(static, sch)
  #define SCHEDULE schedule(auto)
  CODE(3000, 3000)
  #undef DIST_SCHEDULE
  #undef SCHEDULE

  printf("dist_schedule(static,dssch), schedule(guided, sch)\n");
  #define DIST_SCHEDULE dist_schedule(static, dssch)
  #define SCHEDULE schedule(guided, sch)
  CODE(3000, 3000)
  #undef DIST_SCHEDULE
  #undef SCHEDULE

  #pragma omp target exit data map(release:a[:MAX_N],b[:MAX_N],c[:MAX_N])

  return 0;
}

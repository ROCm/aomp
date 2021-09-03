#include <stdio.h>
#include <omp.h>

int main() {
//
//  fast_fp_atomics: test functionality of fast FP atomics on MI200 and
//  shows performance comparison
//
  int main_rc = 0;
  int64_t N[] = {5001,50001,50000001}; // 5000000001 fails both fast FP atomics and reductions
  double expectd[] = {(double) (((double)N[0]-1)*(double)N[0])/2.0,
		     (double) (((double)N[1]-1)*(double)N[1])/2.0,
		     (double) (((double)N[2]-1)*(double)N[2])/2.0,
		     (double) (((double)N[3]-1)*(double)N[3])/2.0};
  float expectf[] = {(float) (((float)N[0]-1)*(float)N[0])/2.0f,
		     (float) (((float)N[1]-1)*(float)N[1])/2.0f,
		     (float) (((float)N[2]-1)*(float)N[2])/2.0f,
		     (float) (((float)N[3]-1)*(float)N[3])/2.0f};

  // First tests, functionality (N has to be low otherwise the CAS loop might run forever)
  for(int tt = 0; tt < 1; tt++) {
    double a = 0.0;
    float b = 0.0f;
    int64_t n = N[tt];

    printf("---------------------------------\n");
    printf("Functionality tests for N = %ld\n", n);
    printf("---------------------------------\n");
    fflush(stdout);

    // CAS loops atomic increment's double and single precision
    double t_double_cas0 = omp_get_wtime();
    #pragma omp target teams distribute parallel for map(tofrom:a)
    for(int64_t ii = 0; ii < n; ++ii) {
      #pragma omp atomic hint(amd_safe_fp_atomics)
      a+=(double)ii;;
    }
    double t_double_cas1 = omp_get_wtime()-t_double_cas0;
    if (a == expectd[tt]) {
      printf("Success atomic sum of %ld double's using CAS loop is: %f in %f secs\n",N[tt],a,t_double_cas1);
    } else {
      printf("FAIL ATOMIC SUM N:%ld result: %f != expect: %f \n", N[tt],a,expectd[tt]);
      main_rc=1;
    }

    double t_single_cas0 = omp_get_wtime();
    #pragma omp target teams distribute parallel for map(tofrom:b)
    for(int64_t ii = 0; ii < n; ++ii) {
      #pragma omp atomic hint(amd_safe_fp_atomics)
      b+=(float)ii;;
    }
    double t_single_cas1 = omp_get_wtime()-t_single_cas0;
    if (b == expectf[tt]) {
      printf("Success atomic sum of %ld double's using CAS loop is: %f in %f secs\n",N[tt],a,t_single_cas1);
    } else {
      printf("FAIL ATOMIC SUM N:%ld result: %f != expect: %f \n", N[tt],a,expectf[tt]);
      main_rc=1;
    }

    // Fast FP atomic increment's double and single precision
    a = 0.0;
    b = 0.0f;
    double t_double_fast0 = omp_get_wtime();
    #pragma omp target teams distribute parallel for map(tofrom:a)
    for(int64_t ii = 0; ii < n; ++ii) {
      #pragma omp atomic hint(amd_fast_fp_atomics)
      a+=(double)ii;;
    }
    double t_double_fast1 = omp_get_wtime()-t_double_fast0;
    if (a == expectd[tt]) {
    printf("Success atomic sum of %ld double's using fast FP atomics is: %f in %f secs\n",N[tt],a,t_double_fast1);
    } else {
      printf("FAIL ATOMIC SUM N:%ld result: %f != expect: %f \n", N[tt],a,expectd[tt]);
      main_rc=1;
    }

    #if 0 // llvm back end stack trace
    double t_single_fast0 = omp_get_wtime();
    #pragma omp target teams distribute parallel for map(tofrom:b)
    for(int64_t ii = 0; ii < n; ++ii) {
      #pragma omp atomic hint(amd_fast_fp_atomics)
      b+=(float)ii;;
    }
    double t_single_fast1 = omp_get_wtime()-t_single_fast0;
    if (b == expectf[tt]) {
      printf("Success atomic sum of %ld double's using fast FP atomics is: %f in %f secs\n",N[tt],a,t_single_fast1);
    } else {
      printf("FAIL ATOMIC SUM N:%ld result: %f != expect: %f \n", N[tt],a,expectf[tt]);
      main_rc=1;
    }
    #endif // llvm back end stack trace

    // Time OpenMP reductions for performance comparison, double and single precision
    double ra_double = 0.0;
    double t_red_double_begin = omp_get_wtime();
    #pragma omp target teams distribute parallel for reduction(+:ra_double)
    for(int64_t ii = 0; ii < n; ++ii) {
      ra_double+=(double)ii;;
    }
    double t_red_double_end = omp_get_wtime() - t_red_double_begin;

    if (ra_double == expectd[tt]) {
      printf("Success reduction sum of %ld double's is: %f in %f secs\n",N[tt],ra_double,t_red_double_end);
    } else {
      printf("FAIL REDUCTION SUM N:%ld result: %f != expect: %f \n", N[tt],ra_double,expectd[tt]);
      main_rc=1;
    }

    #if 0 // single precision reductions fail on mi200
    float ra_single = 0.0f;
    double t_red_single_begin = omp_get_wtime();
    #pragma omp target teams distribute parallel for reduction(+:ra_single)
    for(int64_t ii = 0; ii < n; ++ii) {
      ra_single+=(float)ii;;
    }
    double t_red_single_end = omp_get_wtime() - t_red_single_begin;

    if (ra_single == expectf[tt]) {
      printf("Success reduction sum of %ld float's is: %f in %f secs\n",N[tt],ra_single,t_red_single_end);
    } else {
      printf("FAIL REDUCTION SUM N:%ld result: %f != expect: %f \n", N[tt],ra_single,expectf[tt]);
      main_rc=1;
    }
    #endif
  }

  // Then, test performance difference between fast FP atomics and reductions by varying N
  printf("-----------------\n");
  printf("Performance tests\n");
  printf("-----------------\n");

  for(int tt = 0; tt < sizeof(N)/sizeof(*N); tt++) {
    double a = 0.0;
    float b = 0.0f;
    int64_t n = N[tt];

    printf("N = %ld\n", n);
    fflush(stdout);

    // Fast FP atomic increment's double and single precision
    double t_double_fast0 = omp_get_wtime();
    #pragma omp target teams distribute parallel for map(tofrom:a)
    for(int64_t ii = 0; ii < n; ++ii) {
      #pragma omp atomic hint(amd_fast_fp_atomics)
      a+=(double)ii;;
    }
    double t_double_fast1 = omp_get_wtime()-t_double_fast0;
    if (a == expectd[tt]) {
    printf("Success atomic sum of %ld double's using fast FP atomics is: %f in %f secs\n",N[tt],a,t_double_fast1);
    } else {
      printf("FAIL ATOMIC SUM N:%ld result: %f != expect: %f \n", N[tt],a,expectd[tt]);
      main_rc=1;
    }

    // Time OpenMP reductions for performance comparison, double and single precision
    double ra_double = 0.0;
    double t_red_double_begin = omp_get_wtime();
    #pragma omp target teams distribute parallel for reduction(+:ra_double)
    for(int64_t ii = 0; ii < n; ++ii) {
      ra_double+=(double)ii;;
    }
    double t_red_double_end = omp_get_wtime() - t_red_double_begin;

    if (ra_double == expectd[tt]) {
      printf("Success reduction sum of %ld double's is: %f in %f secs\n",N[tt],ra_double,t_red_double_end);
    } else {
      printf("FAIL REDUCTION SUM N:%ld result: %f != expect: %f \n", N[tt],ra_double,expectd[tt]);
      main_rc=1;
    }
  }

  return main_rc;
}

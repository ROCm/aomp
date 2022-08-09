#include <stdio.h>
#include <omp.h>
int main() {
//
//  atomic_vs_reduction.c: This test shows how much faster reductions are than atomic operations 
//
  int main_rc = 0;
  int N       = 5001;
  float expect = (float) (((float)N-1)*(float)N)/2.0;

  for (int i = 0; i < 4; i++) {
    #pragma omp target
    {
    }
  }

  float a    = 0.0;
  double t0 = omp_get_wtime();
  #pragma omp target teams distribute parallel for map(tofrom:a)
  for(int ii = 0; ii < N; ++ii) {
    #pragma omp atomic hint(AMD_fast_fp_atomics)
    a+=(float)ii;
  }
  double t1 = omp_get_wtime()-t0;
  if (a == expect) {
    printf("Success atomic with hint (fast FP atomic) sum of %d integers is: %f in \t\t%f secs\n",N,a,t1);
  } else {
    printf("FAIL ATOMIC SUM N:%d result: %f != expect: %f \n", N,a,expect);
    main_rc=1;
  }

  float casa    = 0.0;
  double t_cas0 = omp_get_wtime();
  #pragma omp target teams distribute parallel for map(tofrom:casa)
  for(int ii = 0; ii < N; ++ii) {
    #pragma omp atomic
    casa+=(float)ii;
  }
  double t_cas1 = omp_get_wtime()-t_cas0;
  if (casa == expect) {
    printf("Success atomic without hint (cas loop) sum of %d integers is: %f in \t\t%f secs\n",N,casa,t_cas1);
  } else {
    printf("FAIL ATOMIC SUM N:%d result: %f != expect: %f \n", N,casa,expect);
    main_rc=1;
  }

  // Now do the sum as a reduction
  float ra = 0.0;
  double t2 = omp_get_wtime();
  #pragma omp target teams distribute parallel for reduction(+:ra) 
  for(int ii = 0; ii < N; ++ii) {
    ra+=(float)ii;
  }
  double t3 = omp_get_wtime() - t2;

  if (ra == expect) {
    printf("Success reduction sum of %d integers is: %f in \t\t\t\t\t%f secs\n",N,ra,t3);
  } else {
    printf("FAIL REDUCTION SUM N:%d result: %f != expect: %f \n", N,ra,expect);
    main_rc=1;
  }
  return main_rc;
}

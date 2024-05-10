#include <stdio.h>
#include "assert.h"
#include <unistd.h>
#include <inttypes.h>

#define NUM_THREADS 256

#define _INLINE_ATTR_ __attribute__((flatten, always_inline))
#define _RF_LDS volatile __attribute__((address_space(3)))
#if defined(__AMDGCN__) || defined(__NVPTX__)
void __kmpc_rfun_sum_d(double *val, double otherval);
void __kmpc_rfun_sum_lds_d(_RF_LDS double *val, _RF_LDS double *otherval);
void _INLINE_ATTR_  __kmpc_iteamr_d_16x64
   (double v, double *r_ptr, void (*_rf)(double *, double),
      void (*_rf_lds)(_RF_LDS double *, _RF_LDS double *), const double iv, const uint64_t k);
#else
void __kmpc_rfun_sum_d(double *val, double otherval) {}
void __kmpc_rfun_sum_lds_d(_RF_LDS double *val, _RF_LDS double *otherval) {}
void _INLINE_ATTR_  __kmpc_iteamr_d_16x64
   (double v, double *r_ptr, void (*_rf)(double *, double),
    void (*_rf_lds)(_RF_LDS double *, _RF_LDS double *), const double iv, const uint64_t k) {}
#endif

void vmul_omp(double*a, double*b, double*c, int N) {
#pragma omp target map(to: a[0:N], b[0:N]) map(from:c[0:N]) 
#pragma omp teams distribute
  for(int i=0; i<N; i++) {
    double sum = 0;
#pragma omp parallel for reduction(+:sum)
    for(int j=0; j<N; j++) {
      sum += a[i] * b[j];
    }
    c[i] = sum;
  }
}

void vmul_sim(double*a, double*b, double*c, int N) {
#pragma omp target map(to: a[0:N], b[0:N]) map(from:c[0:N]) 
#pragma omp teams distribute
  for(int i=0; i<N; i++) {
    double sum = 0;
#pragma omp parallel for num_threads(NUM_THREADS)
    for (uint64_t k = 0; k < NUM_THREADS; ++k) {
      double val0 = 0;
      for (int64_t j = k; j < N; j += NUM_THREADS)
	val0 += a[i] * b[j];
      __kmpc_iteamr_d_16x64(val0, &sum,
			   __kmpc_rfun_sum_d,
			   __kmpc_rfun_sum_lds_d,
			   0, k);
    }
    c[i] = sum;
  }
}

void reset_c(double *c, int N) {
  for (int i=0; i<N; i++) {
    c[i] = 0;
  }
}

int check(double *c, double *validate, int N, const char *phase) {
  int flag=-1; // Mark Success
  for(int i=0; i<N; i++) {
    if(c[i] != validate[i]) {
      // print 1st bad index
      if( flag == -1 ) 
	printf("First fail: %s: c[%d](%f) != validate[%d](%f)\n",
	       phase, i, c[i], i, validate[i]);
      flag = i;
    }
  }
  if( flag == -1 ){
    printf("Success: %s\n", phase);
    return 0;
  } else {
    printf("Last fail: %s: c[%d](%f) != validate[%d](%f)\n",
	   phase, flag, c[flag], flag, validate[flag]);
    printf("Fail\n");
    return 1;
  }
}

  
int main(){
    const int N = 10000;
    double a[N], b[N], c[N], validate[N];
    for(int i=0; i<N; i++){
      a[i] = i+1;
      double sum = 0;
      for(int j=0; j<N; j++){
        b[j] = j+2;
        sum += a[i] * b[j];
      }
      validate[i] = sum;
    }

    vmul_omp(a, b, c, N);
    int rc_omp = check(c, validate, N, "OMP");
    
    reset_c(c, N);

    vmul_sim(a, b, c, N);
    int rc_sim = check(c, validate, N, "SIM");

    return rc_omp || rc_sim;
}


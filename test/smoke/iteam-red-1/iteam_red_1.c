#include <stdio.h>
#include "assert.h"
#include <unistd.h>
#include <inttypes.h>

#define _INLINE_ATTR_ __attribute__((flatten, always_inline))
#define _RF_LDS volatile __attribute__((address_space(3)))
#if defined(__AMDGCN__) || defined(__NVPTX__)
void __kmpc_rfun_sum_d(double *val, double otherval);
void __kmpc_rfun_sum_lds_d(_RF_LDS double *val, _RF_LDS double *otherval);
void _INLINE_ATTR_  __kmpc_iteamr_d_4x64
   (double v, double *r_ptr, void (*_rf)(double *, double),
      void (*_rf_lds)(_RF_LDS double *, _RF_LDS double *), const double iv, const uint64_t k);
#else
void __kmpc_rfun_sum_d(double *val, double otherval) {}
void __kmpc_rfun_sum_lds_d(_RF_LDS double *val, _RF_LDS double *otherval) {}
void _INLINE_ATTR_  __kmpc_iteamr_d_4x64
   (double v, double *r_ptr, void (*_rf)(double *, double),
    void (*_rf_lds)(_RF_LDS double *, _RF_LDS double *), const double iv, const uint64_t k) {}
#endif

void vmul(double*a, double*b, double*c, int N){
#pragma omp target map(to: a[0:N],b[0:N]) map(from:c[0:N]) 
#pragma omp teams distribute num_teams(104)
  for(int i=0;i<N;i++) {
    double sum = 0;
#pragma omp parallel for reduction(+:sum) num_threads(256)
    //    for(int j=0;j<N;j++) {
    for (uint64_t k = 0; k < 256; ++k) {
      double val0 = 0;
      for (int64_t j = k; j < N; j += 256)
	val0 += a[i]*b[j];
      __kmpc_iteamr_d_4x64(val0, &sum,
			   __kmpc_rfun_sum_d,
			   __kmpc_rfun_sum_lds_d,
			   0, k);
    }
    c[i] = sum;
  }
}

int main(){
    const int N = 100000;
    double a[N],b[N],c[N],validate[N];
    int flag=-1; // Mark Success
    for(int i=0;i<N;i++){
      a[i]=i+1;
      double sum = 0;
      for(int j=0;j<N;j++){
        b[j]=j+2;
        sum += a[i]*b[j];
      }
      validate[i] = sum;
    }

    vmul(a,b,c,N);

    for(int i=0;i<N;i++) {
        if(c[i]!=validate[i]) {
//          print 1st bad index
            if( flag == -1 ) 
              printf("First fail: c[%d](%f) != validate[%d](%f)\n",i,c[i],i,validate[i]);
            flag = i;
        }
    }
    if( flag == -1 ){
      printf("Success\n");
        return 0;
    } else {
        printf("Last fail: c[%d](%f) != validate[%d](%f)\n",flag,c[flag],flag,validate[flag]);
        printf("Fail\n");
        return 1;
    }
}


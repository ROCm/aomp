#include <time.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define N 1000
#define Eps 1e-7

#pragma omp declare target
void func_1v(float*, float*, unsigned);
void func_2v(float*, float*, unsigned);
void func_3v(float*, float*, unsigned);
#pragma omp end declare target

int main(){
  float a[N], t1[N], t2[N], s = 0;
  unsigned i;
  unsigned nErr = 0;

  srand((unsigned int)time(NULL));
  #pragma omp parallel for
  for(i=0; i<N; ++i){
    a[i]=rand()%100;
  }

  func_1v(a,t1,N);
  func_3v(a,t2,N);

  #pragma omp parallel for reduction(+:s)
  for(i=0; i<N; ++i) s += t1[i];
  if(s < Eps){
    printf("Check 0: All elemets are zeros!\n");
    return -1;
  }

  for(i=0; i<N; ++i){
    if(fabs(t1[i]-t2[i]) >= Eps){
      ++nErr;
      printf("Check 1: error at %d: %e >= %e\n",i,fabs(t1[i]-t2[i]),Eps);
    }
  }

  func_2v(t1,t2,N);

  for(i=0; i<N; ++i){
    if(fabs(a[i]-t2[i]) >= Eps){
      ++nErr;
      printf("Check 2: error at %d: %e >= %e\n",i,fabs(a[i]-t2[i]),Eps);
    }
  }

  if(!nErr) printf("Success\n");

  return nErr;
}

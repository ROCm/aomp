#include <stdio.h>

#include "../utilities/check.h"

#define N 100

int test_aligned(){
  int a[N], aa[N];
  int i, error = 0;

  // initialize
  for(i=0; i<N; i++)
    aa[i] = a[i] = -1;

  int *b = a;

  // offload
  #pragma omp target simd map(tofrom: b[0:100]) aligned(b: 8*sizeof(int))
  for(int k=0; k<N; k++)
    b[k] = k;

  // host
  for(i=0; i<N; i++)
    aa[i] = i;

  // check
  for(i=0; i<N; i++) {
    if (a[i] != aa[i]) 
      printf("%d: a %d != %d (error %d)\n", i, a[i], aa[i], ++error);
    if (error > 10) {
      printf("abort\n");
      return error;
    }
  }
  return error;
}

int test_collapsed(){
  int a[N], aa[N];
  int i, error = 0;

  // initialize
  for(i=0; i<N; i++)
    aa[i] = a[i] = -1;

  // offload
  #pragma omp target simd map(tofrom: a[0:100]) collapse(2) 
  for(int k=0; k<N/4; k++)
    for(int l=0; l<4; l++)
      a[k*4+l] = k*4+l;

  // host
  for(i=0; i<N; i++)
    aa[i] = i;

  // check
  for(i=0; i<N; i++) {
    if (a[i] != aa[i]) 
      printf("%d: a %d != %d (error %d)\n", i, a[i], aa[i], ++error);
    if (error > 10) {
      printf("abort\n");
      return error;
    }
  }
  return error;
}
#if 0
int test_lastprivate(){
  // TODO
  int a[N], aa[N];
  int i, error = 0;

  // initialize
  for(i=0; i<N; i++)
    aa[i] = a[i] = -1;

  int n;

  // offload
  #pragma omp target simd map(tofrom: a[0:100]) lastprivate(n)
  for(int k=0; k<N; k++) {
    a[k] = k;
    n = k;
  }
  printf(" n = %d\n", n);
  a[0] = n;

  // host
  for(i=0; i<N; i++)
    aa[i] = i;
  aa[0] = N-1;

  // check
  for(i=0; i<N; i++) {
    if (a[i] != aa[i])
      printf("%d: a %d != %d (error %d)\n", i, a[i], aa[i], ++error);
    if (error > 10) {
      printf("abort\n");
      return error;
    }
  }
  return error;
}
#endif
int test_linear(){
  int a[N], aa[N];
  int i, error = 0;

  // initialize
  for(i=0; i<N; i++)
    aa[i] = a[i] = -1;

  int l = 0;
  // offload
  #pragma omp target simd map(tofrom: a[0:100]) linear(l: 2) 
  for(int k=0; k<N; k++) {
    l = 2*k;
    a[k] = l;
  }

  // host
  for(i=0; i<N; i++)
    aa[i] = 2*i;

  // check
  for(i=0; i<N; i++) {
    if (a[i] != aa[i]) 
      printf("%d: a %d != %d (error %d)\n", i, a[i], aa[i], ++error);
    if (error > 10) {
      printf("abort\n");
      return error;;
    }
  }
  return error;
}


int test_private(){
  int a[N], aa[N];
  int i, error = 0;

  // initialize
  for(i=0; i<N; i++)
    aa[i] = a[i] = -1;

  int n;
  // offload
  #pragma omp target simd map(tofrom: a[0:100]) private(n) 
  for(int k=0; k<N; k++) {
    n = k;
    a[k] = n;
  }

  // host
  for(i=0; i<N; i++)
    aa[i] = i;

  // check
  for(i=0; i<N; i++) {
    if (a[i] != aa[i]) 
      printf("%d: a %d != %d (error %d)\n", i, a[i], aa[i], ++error);
    if (error > 10) {
      printf("abort\n");
      return error;
    }
  }
  return error;
}

int test_safelen(){
  int a[N], aa[N];
  int i, error = 0;

  // initialize
  for(i=0; i<N; i++)
    aa[i] = a[i] = -1;

  // offload
  #pragma omp target simd map(tofrom: a[0:100]) safelen(2) 
  for(int k=0; k<100; k++) {
    if (k > 1){
      a[k] = a[k-2] + 2;
    }
    else{
      a[k] = k;
    }
  }

  // host
  for(i=0; i<N; i++)
    aa[i] = i;

  // check
  for(i=0; i<N; i++) {
    if (a[i] != aa[i]) 
      printf("%d: a %d != %d (error %d)\n", i, a[i], aa[i], ++error);
    if (error > 10) {
      printf("abort\n");
      return error;
    }
  }
  return error;
}

int main()
{
  int error = 0;
  check_offloading();

  // Clauses
  error += test_aligned();
  error += test_collapsed();
  //error += test_lastprivate();
  error += test_linear();
  error += test_private();
  error += test_safelen();

  // report
  printf("done with %d errors\n", error);
  return error;
}

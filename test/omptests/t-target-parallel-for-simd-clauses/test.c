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
  #pragma omp target data map(tofrom: b[0:100]) 
  #pragma omp target parallel for simd aligned(b: 8*sizeof(int))
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
  #pragma omp target data map(tofrom: a[0:100]) 
  {
    #pragma omp target parallel for simd collapse(2)
    for(int k=0; k<N/4; k++)
      for(int l=0; l<4; l++)
        a[k*4+l] = k*4+l;
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

int test_lastprivate(){
  int a[N], aa[N];
  int i, error = 0;

  // initialize
  for(i=0; i<N; i++)
    aa[i] = a[i] = -1;

  int n;
  // offload
  #pragma omp target parallel for simd map(tofrom: a[0:100], n) lastprivate(n)
  for(int k=0; k<N; k++) {
    a[k] = k;
    n = k;
  }
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

int test_linear(){
  int a[N], aa[N];
  int i, error = 0;

  // initialize
  for(i=0; i<N; i++)
    aa[i] = a[i] = -1;

  int l = 0;
  // offload
  #pragma omp target data map(tofrom: a[0:100]) 
  {
    #pragma omp target parallel for simd linear(l: 2)
    for(int k=0; k<N; k++) {
      l = 2*k;
      a[k] = l;
    }
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
  #pragma omp target data map(tofrom: a[0:100]) 
  {
    #pragma omp target parallel for simd private(n)
    for(int k=0; k<N; k++) {
      n = k;
      a[k] = n;
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

int test_safelen(){
  int a[N], aa[N];
  int i, error = 0, k;

  // initialize
  for(i=0; i<N; i++)
    aa[i] = a[i] = -1;

  // offload
  // TODO: Write a better test for safelen
  // Not really a good test for safelen in this case. This works for now.
  #pragma omp target parallel for simd map(tofrom: a[0:100]) schedule(static, 100) safelen(2)
  for(k=0; k<100; k++) {
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
  error += test_lastprivate();
  error += test_linear();
  error += test_private();
  error += test_safelen();

  // report
  printf("done with %d errors\n", error);
  return error;
}

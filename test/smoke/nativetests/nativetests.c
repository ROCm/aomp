#include <stdio.h>
#include <assert.h>

#define POPC_VALUE 0xff00ff00ff000000

// host API
int __nv_popcll(long long a) {
  int i = 0;
  int count = 0;
  while(i<64) {
    if (a & (1ull<<i)) count++;
    i++;
  }
  return count;
}
int test1() {
  int result;
  #pragma omp target map(tofrom:result)
  {
    result = __nv_popcll(POPC_VALUE);
  }

  // verification
  int host = __nv_popcll(POPC_VALUE);
  if (result != host) {
    printf("popcll test failed! result = %d, expecting %d\n", result, host);
    return 1;
  }
  printf("popcll test OK!\n");
  return 0;
}

#define FFS_VALUE 0xf4
// host API
int __nv_ffsll(long long a) {
  int i = 0;
  while(i<64) {
    if (a & (1ull<<i)) break;
    i++;
  }
  if (i==64) return 0; // __nv_ffsll(0)=0
  return i+1; // return 1~64
}
int __nv_ffs(int a) {
  int i = 0;
  while(i<32) {
    if (a & (1<<i)) break;
    i++;
  }
  if (i==32) return 0; // __nv_ffs(0)=0
  return i+1; // return 1~32
}

int test2() {
  int result;
  #pragma omp target map(tofrom:result)
  {
    result = __nv_ffsll(FFS_VALUE);
  }

  // verification
  int host = __nv_ffsll(FFS_VALUE);
  if (result != host) {
    printf("ffsll test failed! result = %d, expecting %d\n", result, host);
    return 1;
  }
  printf("ffsll test OK!\n");


  #pragma omp target map(tofrom:result)
  {
    result = __nv_ffs(FFS_VALUE);
  }

  // verification
  host = __nv_ffs(FFS_VALUE);
  if (result != host) {
    printf("ffs test failed! result = %d, expecting %d\n", result, host);
    return 1;
  }
  printf("ffs test OK!\n");

  return 0;
}


// __ballot64 test is AMDGCN specific test only
#if 0
#define N_THREADS 64
//host API
long long __ballot64(int a) {
  // max num is 64 threads in a warp
  assert (N_THREADS <= 64 && N_THREADS > 0);
  if (N_THREADS == 64) return -1; // 0xffff ffff ffff ffff
  // 64-bit mask
  return (1ull<<N_THREADS) - 1;
}

int test3() {
  long long result;
  #pragma omp target parallel num_threads(N_THREADS) map(tofrom:result)
  #pragma omp single
  {
    result = __ballot64(1);
  }

  // verification
  long long host = __ballot64(1);
  if (result != host) {
    printf("ballot64 test failed! result = %lld, expecting %lld\n", result, host);
    return 1;
  }
  printf("ballot64 test OK!\n");

  return 0;
}
#else
int test3() { return 0;}
#endif

#if 1
int test4() {
#define SHIFTNUM 60
  unsigned long long result = 1;
  #pragma omp target parallel map(tofrom:result)
  #pragma omp single
  {
    result = result << SHIFTNUM;
  }

  if (result != 1ll << SHIFTNUM) {
    printf("shift result = %llx\n", result);
    return 1;
  }
  printf("Shift test OK!\n");

  return 0;
}
#else
int test4() { return 0;}
#endif

int main () {

  int error = 0;
  error += test1();
  error += test2();
  error += test3();
  error += test4();


  if (!error)
    printf("native instructions test OK\n");
  else 
    printf("native instructions test failed\n");

  return error;
}

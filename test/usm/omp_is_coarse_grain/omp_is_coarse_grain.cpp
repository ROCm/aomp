#include <stdlib.h>
#include <stdio.h>
#include <malloc.h>

#include <omp.h>

#define N 10000

#define PAGE_SIZE 4096

//#pragma omp requires unified_shared_memory

int main() {
  int n = N;
  int *a = (int *) malloc(n*sizeof(int));
  int *b = (int *) malloc(n*sizeof(int));

  if(!a || !b) return 1;

  // not mapped yet, not coarse grain
  bool ret = !omp_is_coarse_grain_mem_region(a, n*sizeof(int));

  #pragma omp target enter data map(to:a[:n])

  // a was mapped above, it must be coarse grain
  ret |= omp_is_coarse_grain_mem_region(a, n*sizeof(int));

  // going beyond a's mapped area, not coarse grain
  ret |= !omp_is_coarse_grain_mem_region(a, 2*n*sizeof(int));

  #pragma omp target
  {}

  #pragma omp target exit data map(delete:a[:n])

  // a continues being coarse grain even after it has been deleted from
  // maps
  ret |= omp_is_coarse_grain_mem_region(a, n*sizeof(int));

  // not mapped yet, not coarse grain
  ret |= !omp_is_coarse_grain_mem_region(b, n*sizeof(int));

  #pragma omp target data map(tofrom: b[:n])
  {
    // b mapped since start of current topmost synctactic region
    // where it was made coarse grain
    ret |= omp_is_coarse_grain_mem_region(b, n*sizeof(int));

    // going beyond b's mapped area, not coarse grain
    ret |= !omp_is_coarse_grain_mem_region(b, 2*n*sizeof(int));

    #pragma omp target
    {}
  }

  // b stays coarse grain also after target data synctactic region
  ret |= omp_is_coarse_grain_mem_region(b, n*sizeof(int));

  return !ret;
}

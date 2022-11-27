#include <omp.h>
#include <stdio.h>
#include <stdlib.h>

#define N 8

//int x;
int main() {

#pragma omp target parallel for num_threads(8) //uses_allocators(omp_pteam_mem_alloc) allocate(omp_pteam_mem_alloc: x) private(x) 
  for (int i = 0; i < N; i++) {
    printf("%d\n", omp_get_thread_num());
  }
  printf("should print 0-7\n");
  return 0;
}

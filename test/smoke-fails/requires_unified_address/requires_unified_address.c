// Based on sollve_vv/tests/5.0/requires/test_requires_unified_address.c

#include <stdio.h>
#include <omp.h>
#define N 1024

#pragma omp requires unified_address

int unified_address() {

   int errors = 0;
   int i;
   int * mem_ptr = (int *)malloc(N * sizeof(int));

   #pragma omp target map(to: mem_ptr)
   {
      for (i = 0; i < N; i++) {
         mem_ptr[i] = i + 1;
      }
   }
   
   for (i = 0; i < N; i++) {
      if(mem_ptr[i] != i + 1) {
         errors++;
      }  
   }
   
   return errors;
}

int main(void) {

  if (unified_address()) {
    return -1;
  }

  printf("Success\n");
  return 0;
}

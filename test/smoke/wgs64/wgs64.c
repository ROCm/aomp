#include <stdio.h>
#include <omp.h>

int n =64;

int main(void) {

  int fail = 0;
  int a = -1;
  //
#if 1
#pragma omp target
  { //nothing
  }
#endif
  #pragma omp target teams distribute  thread_limit(64)
  for (int k =0; k < n; k++)
  {
    // nothing
  }
  #pragma omp target teams distribute  parallel for thread_limit(64)
  for (int k =0; k < n; k++)
  {
    // nothing
  }
  printf("Succeeded\n");

  return fail;
}


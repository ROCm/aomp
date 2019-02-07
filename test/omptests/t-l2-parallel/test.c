
#include <stdio.h>
#include <omp.h>

#include "../utilities/check.h"

int main(void) {
  check_offloading();

  int A[32][10];

  for (int i=0;i<32;++i)
    for (int j=0; j<10;++j)
      A[i][j] = 0;

  #pragma omp target
  {
    printf ("Serial\n");
    #pragma omp parallel num_threads(32)
    {
      int tt = omp_get_thread_num();
      if (tt == 1) A[tt][0] = 99;
      if (tt % 4 == 0)
      #pragma omp parallel for
      for (int i = 0; i < 10; i++) {
        A[tt][i] += 3;
      }
    }
  }

  for (int i=0;i<32;++i)
    for (int j=0; j<10;++j)
      printf("L1:%d L2:%d %d\n",i,j,A[i][j]);

  printf("Done!");
  return 0;
}

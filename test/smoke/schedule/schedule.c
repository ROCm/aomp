#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

int ordered_example(int lb, int ub, int stride)
{
  int i;
  int size = (ub-lb)/ stride;
  double *output = (double*)malloc(size * sizeof(double));

  #pragma omp target teams map(from:output[0:size])
  #pragma omp parallel for ordered schedule(dynamic)
  for (i=lb; i<ub; i+=stride) {
    #pragma omp ordered
    {
      ///////////////////////////////////////
      //
      // Make sure device printf is available, otherwise freezing
      printf(" %d\n", i);
      // The following shall be printed in order
      // 0
      // 5
      // 10
      // 15
      // 20
      // 21
      // 22
      //..
      // 95
      //
      ////////////////////////////////////////
      output[(i-lb)/stride] = omp_get_wtime();
    }
  }

  // verification
  for (int j = 0; j < size; j++) {
    for (int jj = j+1; jj < size; jj++) {
      if (output[j] > output[jj]) {
        printf("Fail to schedule in order.\n");
        free(output);
        return 1;
      }
    }
  }

  free(output);

  printf("test OK\n");

  return 0;
}

int main()
{
  return ordered_example(0, 100, 5);
}


#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

int ordered_example(int lb, int ub, int stride, int nteams) {
  int i;
  int size = (ub-lb)/ stride;
  double *output = (double*)malloc(size * sizeof(double));

#pragma omp target teams map(from                                              \
                             : output [0:size]) num_teams(nteams)              \
    thread_limit(128)
#pragma omp parallel for ordered schedule(dynamic)
  for (i=lb; i<ub; i+=stride) {
    #pragma omp ordered
    {
      ///////////////////////////////////////
      //
      // Make sure device printf is available, otherwise freezing
      printf(" %02d : team %02d of %02d teams : thread %03d of %03d threads\n",
             i, omp_get_team_num(), omp_get_num_teams(), omp_get_thread_num(),
             omp_get_num_threads());
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

// The ordered construct above is executing every iteration on every team
// and on every thread. Every iteration on everythread does not seem correct
// This function is same as above without ordered construct to show
// iterations spread across the threads correctly.  There may be a problem
// with ordered construct.  I believe it should serialize on a single thread.
// This fails on trunk with nvptx as well as on amdgcn.
int NO_order_example(int lb, int ub, int stride, int nteams) {
  int i;
  int size = (ub - lb) / stride;
  double *output = (double *)malloc(size * sizeof(double));

#pragma omp target teams map(from                                              \
                             : output [0:size]) num_teams(nteams)              \
    thread_limit(128)
#pragma omp parallel for
  for (i = lb; i < ub; i += stride) {
    printf(" %02d : team %02d of %02d teams : thread %03d of %03d threads "
           "NO_ORDER\n",
           i, omp_get_team_num(), omp_get_num_teams(), omp_get_thread_num(),
           omp_get_num_threads());
    output[(i - lb) / stride] = omp_get_wtime();
  }

  // verification
  for (int j = 0; j < size; j++) {
    for (int jj = j + 1; jj < size; jj++) {
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
  // use small teamcount=8 to avoid smoke-test timeout due to many synchronous
  // printfs
  NO_order_example(0, 10, 1, 8);
  return ordered_example(0, 10, 1, 8);
}


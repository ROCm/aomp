#include <omp.h>
#include <stdio.h>
#include <stdlib.h>

int main() {
  int max_threads, scalar_num_threads;
  int *num_threads;
  omp_set_num_threads(128);
  // We obtain the default number of threads in the target region
#pragma omp target map(from:max_threads)
  {
    max_threads = omp_get_max_threads();
  }

  num_threads = (int *) malloc (sizeof(int) * max_threads); 
  for (int i = 0; i < max_threads; ++i) {
    num_threads[i] = -1;
  }
#pragma omp target parallel map(from:num_threads[0:max_threads], scalar_num_threads)
  {
#pragma omp master
    {
      scalar_num_threads = omp_get_num_threads();
    }
    int thread_id = omp_get_thread_num();
    num_threads[thread_id] = omp_get_num_threads();
  }
  fprintf(stderr, "MaxThreds %d ScNumThrd %d numThrds %d\n", max_threads, scalar_num_threads, num_threads[0]);
  return 0;
}

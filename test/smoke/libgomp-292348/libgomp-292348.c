/*
  Test hipcc host compilation with -lgomp, from 292348.
*/

#include <stdio.h>
#include <omp.h>

void inc_subarray(int *array, int start, int end) {
  for (int i = start; i <end; i++) {
    array[i] += 1;
  }
}

void inc_subarray_mt(int *array, int start, int end) {
#ifdef _OPENMP
  #pragma omp parallel
  {
    int num_t = omp_get_num_threads();
    int tid = omp_get_thread_num();
    int q = (end - start) / num_t + 1;
    int s = start + tid * q;
    int e = s + q;
    e = (e < end) ? e : end;

    for (int i = s; i < e; i++) {
      array[i] += 1;
    }
    //printf("tid: %d, num_t: %d\n", tid, num_t);
  }
#endif
}

int main(int argc, char *argv[]) {
  int num_threads = 0;
  int errors = 0;

  #pragma omp parallel
  {
    num_threads = omp_get_num_threads();
  }
  printf("omp threads: %d\n", num_threads);
  int ary[1000] = {0};
  int ary_mt[1000] = {0};

  for (int i = 0; i < 1000; i++) {
    ary[i] = 0;
    ary_mt[i] = 0;
  }

  for (int i = 0; i < 10; i++) {
    int start = (i *137) % 1000;
    int end = ((i + 1) *279) % 1000;

    if (start > end) {
      int tmp = start;
      start = end;
      end = tmp;
    }

    printf("%d to %d\n", start, end);
    inc_subarray_mt(ary_mt, start, end);
    inc_subarray(ary, start, end);
  }

  for (int i = 0; i < 100; i++) {
    if (ary_mt[i] != ary[i]) {
      printf("ary[%d]: %d != %d\n", i, ary[i], ary_mt[i]);fflush(stdout);
      errors++;
    }
  }

  if (errors){
    printf("FAIL\n");
    return 1;
  }

    printf("PASS\n");
    return 0;
}

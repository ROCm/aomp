#include <stdio.h>
#include <omp.h>
#define N 1024
#define THREADS 8
int main(void) {

  int errors = 0;
  int total_wait_errors = 0;
  int x[N];
  int y[N];
  int z[N];
  int num_threads = -1;
  int rand_indexes[8];

  for (int i = 0; i < N; i++) {
    x[i] = 1;
    y[i] = i + 1;
    z[i] = 2*(i + 1);
  }

  for (int i = 0; i < THREADS; i++) {
    rand_indexes[i] = rand()%(N + 1);
  }

  #pragma omp target parallel num_threads(THREADS) map(tofrom: x[0:N], num_threads, total_wait_errors) map(to: y[0:N], z[0:N])
  {
    #pragma omp loop order(unconstrained:concurrent)
    for (int i = 0; i < N; i++) {
      x[i] += y[i]*z[i];
    }
    if (x[rand_indexes[omp_get_thread_num()]] == 1) {
    #pragma omp atomic update
      total_wait_errors++;
    }
    if (omp_get_thread_num() == 0) {
      num_threads = omp_get_num_threads();
    }
  }

  for (int i = 0; i < N; i++) {
    if(x[i] != 1 + (y[i]*z[i]))
      errors++;
  }
  if(errors || total_wait_errors){
    printf("Fail!");
    return 1;
  }
  printf("Success!");
  return 0;

}


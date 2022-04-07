#include <omp.h>
#include <stdio.h>
int main(){
    omp_set_default_device(0);
    int N=10;
    double *x = (double*) omp_target_alloc(N*sizeof(double),omp_get_default_device());
#pragma omp target enter data map(to: x[:N])
    fprintf(stderr, "before target data: x[1] 0x%p \n",&x[1]);
    #pragma omp target data use_device_ptr(x)
    {
      fprintf(stderr, "in target data: x[1] 0x%p \n",&x[1]);
    }
    #pragma omp target
    {
      for (int i = 0; i < N; ++i) x[i] = 2.0;
      printf("in target x[1] 0x%p \n",&x[1]);
    }

  return 0;
}

#include <omp.h>
#include <stdio.h>
int main(){
    omp_set_default_device(0);
    int N=10;
    double *x = (double*) omp_target_alloc(N*sizeof(double),omp_get_default_device());
    for (int i = 0; i < N; ++i) x[i] = 2.0;
    fprintf(stderr, "before target data: x[1] 0x%p = %g\n",&x[1], x[1]);
    #pragma omp target data use_device_ptr(x)
    {
      printf("in target data: &x[1] 0x%p \n",&x[1]);
      printf("in target data: x[1] %g\n", x[1]);
    }

  return 0;
}

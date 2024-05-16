#include <omp.h>
#include <stdio.h>

#pragma omp requires unified_shared_memory

int main(){
    int N;
    double *x;
    N = 128;
    x = new double[N];

    for (int i = 0; i < N; ++i) x[i] = 1.0;
    printf("initial on host                  : x[1] = %g addr:%p\n",x[1],(void*) &x[1]);

    #pragma omp target enter data  map(to:x[0:N])
    printf("after enter data                 : x[1] = %g addr:%p\n",x[1],(void*) &x[1]);

    for (int i = 0; i < N; ++i) x[i] = 2.0;

    printf("after initialize to 2            : x[1] = %g addr:%p\n",x[1],(void*) &x[1]);
    #pragma omp target data use_device_ptr(x)
    {
      printf("in target data w/use_device_ptr  : x[1] = NA addr:%p\n",(void*) &x[1]);
      #pragma omp target
        printf("in target region w/use_device_ptr: x[1] = NA addr:%p\n",(void*) &x[1]);
        printf("after  target region             : x[1] = NA addr:%p\n",(void*) &x[1]);
    }
    printf("outside target data              : x[1] = %g addr:%p\n",x[1],(void*) &x[1]);
    #pragma omp target exit data map(release:x[0:N])
    printf("after exit target data           : x[1] = %g addr:%p\n",x[1],(void*) &x[1]);

    delete[] x;
    return 0;

}

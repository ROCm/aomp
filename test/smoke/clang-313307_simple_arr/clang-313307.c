#include <stdio.h>
#include <omp.h>

#define SIMPLE_SPMD 0
int main(){
    int nthreads_a[3];
    int a1_a[4];
#pragma omp target parallel map(tofrom: a1_a, nthreads_a)
{
    a1_a[0] = 1;
    nthreads_a[0] = 4;
}

    printf("hello %d %d\n", a1_a[0], nthreads_a[0]);

    if (a1_a[0] != 1 || nthreads_a[0] != 4) return 1; 
    return 0;
}


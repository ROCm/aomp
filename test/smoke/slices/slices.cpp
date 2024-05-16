#include <stdio.h>
#include <stdlib.h>
const int N = 10; 
void foo(double *A) {
#pragma omp target
    {   
        A[0] = 1.0;
    }   
}
int main() {
    double *A = (double *)malloc(sizeof(double) * N); 
    for (int i = 0; i < N; i++) {
        A[i] = 0.0;
    }   
#pragma omp target data map(tofrom:A[0 : N])
    {   
        foo(&A[5]);
    }   
    for (int i = 0; i < N; i++) {
        printf("%f\n", A[i]);
    }   
}

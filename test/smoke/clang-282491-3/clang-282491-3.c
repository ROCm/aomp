#include <stdlib.h>
#include <stdio.h>
#include <omp.h>
#define N 100
typedef struct myvec{
    size_t len;
    double *data;
} myvec_t;

int main(){
    myvec_t s;
    s.data = (double *)calloc(N,sizeof(double));
    if(!s.data){
        fprintf(stderr, "alloc failed\n");
        exit(1);
    }
    s.len = N;
    printf("CPU: Array at %p with length %zu\n", s.data, s.len);
    #pragma omp target map(s)
    printf("GPU: Array at %p with length %zu\n", s.data, s.len);
}

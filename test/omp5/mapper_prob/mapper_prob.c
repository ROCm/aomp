#include <stdlib.h>
#include <stdio.h>
#include <omp.h>
#define N 100
typedef struct myvec{
    size_t len;
    double *data;
} myvec_t;

#pragma omp declare mapper(myvec_t v) \
    map(v, v.data[0:v.len])
void init(myvec_t *s);

int main(){
    myvec_t s;
    s.data = (double *)omp_target_alloc(N * sizeof(double), /*device: */0);
    s.len = N;
    #pragma omp target map(s)
    init(&s);
    
    printf("s.data[%d]=%lf\n",N-1,s.data[N-1]);
    omp_target_free(s.data, /*Device: */0);
    return 0;
}
void init(myvec_t *s)
{ for(int i=0; i<s->len; i++) s->data[i]=i; }


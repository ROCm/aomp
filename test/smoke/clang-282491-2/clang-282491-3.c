#include <stdlib.h>
#include <stdio.h>
#include <omp.h>
#define N 100
typedef struct myvec{
    size_t len;
    double *data;
} myvec_t;

void init(myvec_t *s);

int main(){
    myvec_t s;
    s.data = (double *)calloc(N,sizeof(double));
    s.len = N;
    #pragma omp target map(s, s.data[:s.len])
    init(&s);
    
    printf("s.data[%d]=%lf\n",N-1,s.data[N-1]);
}

void init(myvec_t *s)
{ for(int i=0; i<s->len; i++) s->data[i]=i; }

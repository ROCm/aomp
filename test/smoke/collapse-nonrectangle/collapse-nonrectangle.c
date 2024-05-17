#include <stdio.h>
#include "assert.h"
#include <unistd.h>

void vmul(int*a, int*b, int*c, int N){
#pragma omp target map(to: a[0:N],b[0:N]) map(from:c[0:N*N])
#pragma omp teams distribute parallel for collapse(2)
   for(int i=0;i<N;i++)
     for(int j=0;j<N-i;j++){
      c[i*N+j]=a[i]*b[j];
   }
}

int main(){
    const int N = 1000;
    int a[N],b[N],c[N*N],validate[N*N];
    int flag=-1; // Mark Success
    for(int i=0;i<N;i++){
      a[i]=i+1;
      for(int j=0;j<N-i;j++){
        b[j]=j+2;
        validate[i*N+j]=a[i]*b[j];
      }
    }

    vmul(a,b,c,N);

    for(int i=0;i<N;i++){
      for(int j=0;j<N-i;j++){
            if( validate[i*N+j] != a[i]*b[j])
              printf("First fail: c[%d](%d) != validate[%d](%d)\n",i,c[i],i,validate[i]);
            flag = i;
        }
    }
    return 0;
}


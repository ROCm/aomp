#include <stdio.h>
#include "assert.h"
#include <unistd.h>

void vmul(int*a, int*b, int*c, int N){
#pragma omp target map(to: a[0:N],b[0:N]) map(from:c[0:N*N])
//#pragma omp teams distribute parallel for collapse(2)
//#pragma omp teams distribute collapse(2)
#pragma omp target teams loop collapse(2)
   for(int i=0;i<N;i++)
     for(int j=0;j<N;j++){
      c[i*N+j]=a[i]*b[j];
   }
}

int main(){
    const int N = 1000;
    int a[N],b[N],c[N*N],validate[N*N];
    int flag=-1; // Mark Success
    for(int i=0;i<N;i++){
      a[i]=i+1;
      for(int j=0;j<N;j++){
        b[j]=j+2;
        validate[i*N+j]=a[i]*b[j];
      }
    }

    vmul(a,b,c,N);

    for(int i=0;i<N*N;i++) {
        if(c[i]!=validate[i]) {
//          print 1st bad index
            if( flag == -1 ) 
              printf("First fail: c[%d](%d) != validate[%d](%d)\n",i,c[i],i,validate[i]);
            flag = i;
        }
    }
    if( flag == -1 ){
        printf("Success\n");
        return 0;
    } else {
        printf("Last fail: c[%d](%d) != validate[%d](%d)\n",flag,c[flag],flag,validate[flag]);
        printf("Fail\n");
        return 1;
    }
}


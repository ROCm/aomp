#include <stdio.h>
#include "assert.h"
#include <unistd.h>

void vmul(int*a, int*b, int*c, int N){
#pragma omp target map(to: a[0:N],b[0:N]) map(from:c[0:N])
//#pragma omp teams distribute parallel for collapse(2)
#pragma omp teams distribute
  for(int i=0;i<N;i++) {
    int sum = 0;
#pragma omp parallel for reduction(+:sum)
    for(int j=0;j<N;j++) {
      sum += a[i]*b[j];
    }
    c[i] = sum;
  }
}

int main(){
    const int N = 100;
    int a[N],b[N],c[N],validate[N];
    int flag=-1; // Mark Success
    for(int i=0;i<N;i++){
      a[i]=i+1;
      int sum = 0;
      for(int j=0;j<N;j++){
        b[j]=j+2;
        sum += a[i]*b[j];
      }
      validate[i] = sum;
    }

    vmul(a,b,c,N);

    for(int i=0;i<N;i++) {
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


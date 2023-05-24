#include <stdio.h>
#include "assert.h"
#include <unistd.h>

void vmul(double*a, double*b, double*c, int N){
#pragma omp target map(to: a[0:N],b[0:N]) map(from:c[0:N]) 
#pragma omp teams distribute num_teams(104)
  for(int i=0;i<N;i++) {
    double sum = 0;
#pragma omp parallel for reduction(+:sum) num_threads(256)
    for(int j=0;j<N;j++) {
      sum += a[i]*b[j];
    }
    c[i] = sum;
  }
}

int main(){
    const int N = 10000;
    double a[N],b[N],c[N],validate[N];
    int flag=-1; // Mark Success
    for(int i=0;i<N;i++){
      a[i]=i+1;
      double sum = 0;
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
              printf("First fail: c[%d](%f) != validate[%d](%f)\n",i,c[i],i,validate[i]);
            flag = i;
        }
    }
    if( flag == -1 ){
      printf("Success\n");
        return 0;
    } else {
        printf("Last fail: c[%d](%f) != validate[%d](%f)\n",flag,c[flag],flag,validate[flag]);
        printf("Fail\n");
        return 1;
    }
}


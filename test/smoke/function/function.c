#include <stdio.h>
#include "assert.h"
#include <unistd.h>

#pragma omp declare target
int mul_int(int, int);
#pragma omp end declare target

int mul_int(int a, int b){
  int c;
  c=a*b;
  return c;
}


void vmul(int*a, int*b, int*c, int N){
#pragma omp target map(to: a[0:N],b[0:N]) map(from:c[0:N])
#pragma omp teams distribute parallel for 
   for(int i=0;i<N;i++) {
      c[i]=mul_int(a[i],b[i]);
   }
}

int main(){
    const int N = 100000;
    int a[N],b[N],c[N],validate[N];
    int flag=-1; // Mark Success
    for(int i=0;i<N;i++) {
        a[i]=i+1;
        b[i]=i+2;
        validate[i]=a[i]*b[i];
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


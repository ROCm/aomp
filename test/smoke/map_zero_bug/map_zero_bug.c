#include <stdio.h>
#include "assert.h"
#include <unistd.h>

void vset(int*b, int N){
#pragma omp target map(tofrom: b[0:N])
#pragma omp teams distribute parallel for 
   for(int i=0;i<N;i++) {
      b[i]=i;
   }
}

int main(){
    const int N = 100;    
    int b[N],validate[N];
    int flag=-1; // Mark Success
    for(int i=0;i<N;i++) {
        b[i] = i;
        validate[i]=i;
    }

    vset(b,0);

    for(int i=0;i<N;i++) {
        if(b[i]!=validate[i]) {
//          print 1st bad index
            if( flag == -1 ) 
              printf("First fail: b[%d](%d) != validate[%d](%d)\n",i,b[i],i,validate[i]);
            flag = i;
        }
    }
    if( flag == -1 ){
        printf("Success\n");
        return 0;
    } else {
        printf("Last fail: b[%d](%d) != validate[%d](%d)\n",flag,b[flag],flag,validate[flag]);
        printf("Fail\n");
        return 1;
    }
}


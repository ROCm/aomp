#include <stdio.h>
#include "assert.h"
#include <unistd.h>

// Select which VMUL[0123] to use
#define VMUL0

#ifdef VMUL0

void vmul(int*a, int*b, int*c, int N){
#pragma omp target map(to: a[0:N],b[0:N]) map(from:c[0:N])
#pragma omp teams distribute parallel for 
   for(int i=0;i<N;i++) {
      c[i]=a[i]*b[i];
   }
}

#elif defined(VMUL1)

void vmul(int*a, int*b, int*c, int N){
#pragma omp target map(to: a[0:N],b[0:N]) map(from:c[0:N])
   for(int i=0;i<N;i++) {
      c[i]=a[i]*b[i];
   }
}

#elif defined(VMUL2)

void vmul(int*a, int*b, int*c, int N){
#pragma omp target map(to: a[0:N],b[0:N]) map(from:c[0:N])
   {
      c[0]=a[0]*b[0];
   }
}

#elif defined(VMUL3)

void vmul(int*a, int*b, int*c, int N){
   int x = a[0];
   int y = b[0];
   int z;
#pragma omp target map(from:z)
   {
      z = x * y;
   }
   c[0] = z;
}

#else
#error no VMULx defined
#endif

int main(){
    int N = 100000;    
    int a[N],b[N],c[N],validate[N];
    int flag=-1; // Mark Success
    for(int i=0;i<N;i++) {
        a[i]=i+1;
        b[i]=i+2;
        validate[i]=a[i]*b[i];
    }

    vmul(a,b,c,N);

#if defined(VMUL2) || defined(VMUL3)
    N = 1;
#endif
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


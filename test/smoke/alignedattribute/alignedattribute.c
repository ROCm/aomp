#include <stdio.h>

void write_index(int*a, int N){
   int*aptr __attribute__ ((aligned(64))) = a; // THIS FAILS
   // int*aptr = a; // THIS WORKS
   printf(" ===> Encounter target teams distribute par for map tofrom:aptr\n");
#pragma omp target teams distribute parallel for map(tofrom: aptr[0:N])
   for(int i=0;i<N;i++) {
      printf("updating aptr[%d] addr:%p\n",i,&aptr[i]);
      aptr[i]=i;
   }
}

int main(){
    const int N = 10;    
    int a[N],validate[N];
    for(int i=0;i<N;i++) {
        a[i]=0;
        validate[i]=i;
    }

    write_index(a,N);

    int flag=-1; // Mark Success
    for(int i=0;i<N;i++) {
        if(a[i]!=validate[i]) {
//          print 1st bad index
            if( flag == -1 ) 
              printf("First fail: a[%d](%d) != validate[%d](%d)\n",i,a[i],i,validate[i]);
            flag = i;
        }
    }
    if( flag == -1 ){
        printf("Success\n");
        return 0;
    } else {
        printf("Last fail: a[%d](%d) != validate[%d](%d)\n",flag,a[flag],flag,validate[flag]);
        printf("Fail\n");
        return 1;
    }
}


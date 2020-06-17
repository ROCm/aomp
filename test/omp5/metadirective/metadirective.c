#define N 100
#include <stdio.h>

int main()
{
   int errors = 0;
   int v1[N], v2[N], v3[N];
   for(int i=0; i<N; i++){ v1[i]=(i+1); v2[i]=-(i+1); }

   #pragma omp target map(to:v1,v2) map(from:v3) device(0)
   #pragma omp metadirective \
                   when(   device={arch("amdgcn")}: teams loop) \
                   default(                     parallel loop)
     for (int i= 0; i< N; i++)  v3[i] = v1[i] * v2[i];

   printf(" %d  %d\n",v3[0],v3[N-1]); //output: -1  -10000
   for(int i=0; i<N; i++){
     if(v3[i] != v1[i] * v2[i]){
       printf("v3[%d]: %d    Correct:%d\n", i, v3[i], v1[i] * v2[i]);
       errors+=1;
     }
   }

   if(errors){
     printf("Fail!\n");
     return 1;
   }else
     printf("Success\n");

   return 0;
}


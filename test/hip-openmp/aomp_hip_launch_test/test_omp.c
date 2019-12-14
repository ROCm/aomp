#include <stdio.h>
#include <assert.h>
#define N 100
int A[N];
int B[N];

int main() {
   for(int i=0; i<N; i++){
      A[i] =0;
      B[i] =i;
   }
   #pragma omp target map(A,B)
   for(int i=0; i<N; i++){
      A[i] = B[i];
   }

   for(int i=0; i<N; i++){
      assert(A[i] == B[i]);
   }
   printf("PASSED\n");

   return 0;
}

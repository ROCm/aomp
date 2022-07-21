/// Based on OpenMP Spec 5.0 example: Example_target_reverse_offload.7.c
///
/// Expected failure until reverse_offload is supported:
/// --------> output: lld: error: undefined symbol: error_handler
///

#include <stdio.h>
#include <omp.h>

#define N 100

#pragma omp requires reverse_offload

void error_handler(int wrong_value, int index)
{
   printf(" Error in offload: A[%d]=%d\n", index,wrong_value);
   printf("        Expecting: A[i ]=i\n");
// output:  Error in offload: A[99]=-1
//                 Expecting: A[i ]=i

}

// Ensure that error_handler is compiled for host only
#pragma omp declare target device_type(host) to(error_handler)
 
int main()
{
   int A[N];
 
   for (int i=0; i<N; i++) A[i] = i;

   A[N-1]=-1;
 
   #pragma omp target map(A)
   {
      for (int i=0; i<N; i++)
      {
         if (A[i] != i) 
         {
            #pragma omp target device(ancestor: 1) map(always,to: A[i:1])
               error_handler(A[i], i);
         }
      }
   }
   return 0;
}

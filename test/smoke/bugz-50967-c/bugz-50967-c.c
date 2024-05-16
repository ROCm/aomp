#include <stdio.h>
#include <omp.h>

//  This smoke test has shows two problems.
//  The first is compile fail when the subtype has a smaller size than 4 bytes
//  That is both char and short fail. We need to generate compile fail
//  for amdgcn when atomic type is char or short  OR use a temp 4 byte
//  value,  which may not be atomic.
//
//  The 2nd problem is a runtime fail.  num_threads(64) has no control over
//  actual number of threads when the default thread limit is 256.

// Set SUBTYPE to anything equal or greater size than int
// for atomic update not to cause fail in llc.
#define SUBTYPE  char
#define REALTYPE int
int f() {
   REALTYPE b = 0;
   #pragma omp target map(tofrom: b) 
   {
      #pragma omp teams distribute  // thread_limit(64)
      // add clause thread_limit(64) above to circumvent the problem 
      // not getting num_threads 64 in parallel below. 
      // Without thread_limit clause, this incorrectly reports 256
      for(int i = 0; i < 1; ++i) {
         SUBTYPE a = 0;
         #pragma omp parallel num_threads(64)
         {
            #pragma omp atomic update
            a += 1;
         }
         b = (REALTYPE) (a);
      }
   }
   if (b == 64 ) return 0;
   printf("ERROR: expecting 64 got %d\n",b);
   return 1;
}

int main() { return f(); }

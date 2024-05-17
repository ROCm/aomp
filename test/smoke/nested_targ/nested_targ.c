#include <stdio.h>
#include "assert.h"
#include <unistd.h>



    #pragma omp declare target
      int foo();
      #pragma omp declare target 
        int bar();
      #pragma omp end declare target 
    #pragma omp end declare target 
int main() {
  return 0;
}

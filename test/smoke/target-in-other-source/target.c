#include <stdio.h>
#include "assert.h"
#include <unistd.h>

#pragma omp declare target
void vmul(int*a, int*b, int*c, int N){
#pragma omp parallel for 
   for(int i=0;i<N;i++) {
      c[i]=a[i]*b[i];
   }
}
#pragma omp end declare target


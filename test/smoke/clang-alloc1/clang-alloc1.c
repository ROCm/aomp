#include <stdio.h>
#include "assert.h"
#include <unistd.h>

#define NZ 10

int main(){
    int colstat[NZ], i;
    colstat[0]=-1;
#pragma omp target enter data  map(alloc:colstat[0:NZ])
#pragma omp target teams distribute parallel for
    for (i=0; i< NZ; i++) {
      colstat[i] = 2222;
    }
    printf("Success\n");

  return 0;
}


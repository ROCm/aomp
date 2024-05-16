#include <omp.h>
#include <stdio.h>
#pragma omp requires reverse_offload
int main(){
    omp_set_default_device(0);
    int N=10;
    #pragma omp target
    {
      printf("in target\n");
    }

  return 0;
}

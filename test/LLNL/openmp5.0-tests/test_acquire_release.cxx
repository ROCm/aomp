#include <iostream>
#include <stdio.h>
#include <omp.h>
// From OpenMP Examples 5.0.0.

#pragma omp requires atomic_default_mem_order(acq_rel)
int main(int argc, char **argv){

  int x =0, y=0;


#pragma omp parallel shared(x,y)
  {
    int tid=omp_get_thread_num();
    if (tid==0){
      x = 100;
      //#pragma omp flush release // Use this without release below on atomic
#pragma omp atomic write release // This can be seq_cst
      y=1;
    } else{
      int tmp=0;
      while(tmp==0){
#pragma omp atomic read acquire // This can be seq_cst
	tmp = y;
      }
      //#pragma omp flush acquire // Use this without acquire above in atomic
      if (x!=100) printf("test atomic with acquire relase:: FAIL ");
      else  printf("test atomic with acquire release :: PASS \n");
      printf(" tid = %d x= %d \n",tid,x);
    }
  }
}

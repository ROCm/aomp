#include <unistd.h>
#include <sys/time.h>

#include <omp.h>
#include <stdio.h>

#pragma omp requires unified_shared_memory

int main(){

  double * DATA;
  size_t N = 1024/sizeof(double);
  omp_set_default_device(0);
  DATA = (double*)   omp_target_alloc(N*sizeof(double), omp_get_default_device()) ;
  //DATA = new double[N]; 
 
  //fill data on CPU
  #pragma omp parallel for 
  for (size_t i = 0; i < N; ++i)
     DATA[i] = 0.0001*i;

  int iter  = 10;

 //warm up 
  for (int i = 0; i < iter; ++i){
    #pragma omp target teams distribute parallel for is_device_ptr(DATA)
    for (size_t i = 0; i < N; ++i)
       DATA[i] = 0.1;
  }

  double t1,t2;
  t1=omp_get_wtime();
  for (int i = 0; i < iter; ++i){
    #pragma omp target teams distribute parallel for is_device_ptr(DATA) 
    for (size_t i = 0; i < N; ++i)
       DATA[i] = 0.1;
  }
  t2=omp_get_wtime();
  printf("loop time (no  nowait) = %g\n",t2-t1);

  t1=omp_get_wtime();
  for (int i = 0; i < iter; ++i){
    #pragma omp target teams distribute parallel for is_device_ptr(DATA) nowait 
    for (size_t i = 0; i < N; ++i)
       DATA[i] = 0.1;
  }
  // #pragma omp taskwait
  t2=omp_get_wtime();
  printf("loop time (with  nowait) = %g\n",t2-t1);
   #pragma omp taskwait

  for (size_t i = N-10; i < N; ++i) {
    printf("DATA[%zu] = %g\n",i,DATA[i]);
    if (DATA[i] != 0.1) { printf("Failed\n"); return 1; }
  }
  //delete[] DATA;
  
  return 0;
}

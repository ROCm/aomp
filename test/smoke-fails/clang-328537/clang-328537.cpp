#include <omp.h>
#include <stdio.h>

#pragma omp requires unified_shared_memory

int main(){


  size_t N = 1024*1024/sizeof(double);
  double GB = (double) N *sizeof(double) / (1024.0*1024.0*1024.0);
  double *X, *Y; 
  X = new  double[N];
  Y = new  double[N];
  double t1,t2;
 
  

  #pragma omp  parallel for 
  for (size_t i = 0; i < N; ++i){
        X[i] = 0.001*i;
        Y[i] = 0.0003*i;
  }
  
  #pragma omp target enter data map(to:X[0:N],Y[0:N])

  double result_cpu = 0.0;
  double result_cpu_par = 0.0;
  double result_cpu_par_target = 0.0;

  for (size_t i = 0; i < N; ++i)
    result_cpu += X[i]*Y[i];

  for (int iter = 0; iter <  5; ++iter){
    result_cpu_par = 0.0;
    t1 = omp_get_wtime();
    #pragma omp  parallel for reduction(+:result_cpu_par)
    for (size_t i = 0; i < N; ++i)
      result_cpu_par += X[i]*Y[i];
    t2 = omp_get_wtime();
    printf("CPU par : loop time = %g [s] effective read BW = %g [GB/s] \n",t2-t1, 2.0*GB/(t2-t1));
  }

  const int nthreads = 1024;
  int nteams = 440*(1024/nthreads)/2;
  nteams = nteams <= ( (N+nthreads-1)/nthreads) ? nteams : (N+nthreads-1)/nthreads;

  for (int iter = 0; iter <  5; ++iter){
    result_cpu_par_target = 0.0;
    t1 = omp_get_wtime();
    #pragma omp target teams distribute parallel for num_teams(nteams)  thread_limit(1024) map(to:X[0:N],Y[0:N]) map(tofrom:result_cpu_par_target) reduction(+:result_cpu_par_target)
    for (size_t i = 0; i < N; ++i){
       result_cpu_par_target += X[i]*Y[i];
    }
    t2 = omp_get_wtime();
    printf("GPU par : dot loop time = %g [s] effective read BW = %g [GB/s] \n",t2-t1, 2.0*GB/(t2-t1));

    t1 = omp_get_wtime();
    #pragma omp target teams distribute parallel for num_teams(nteams)  thread_limit(1024) map(to:X[0:N],Y[0:N]) 
    for (size_t i = 0; i < N; ++i){
      if ( X[i]*Y[i] < 0.0) X[i] = 0.0;
    }
    t2 = omp_get_wtime();
    printf("GPU par : read loop time = %g [s] effective read BW = %g [GB/s] \n",t2-t1, 2.0*GB/(t2-t1));
  }

  printf("result_cpu = %g \t result_cpu_par = %g \t result_cpu_par_target = %g\n ",result_cpu,result_cpu_par,result_cpu_par_target);

  #pragma omp target exit data map(delete:X[0:N],Y[0:N])

  delete[] X;
  delete[] Y;
  return 0;

}

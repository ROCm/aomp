#include <unistd.h>
#include <sys/time.h>

#include <omp.h>
#include <stdio.h>
#include <stdlib.h>

#include <hip/hip_runtime.h>


#pragma omp requires unified_shared_memory

int main(){


  double * X, * Y, *Z;
  size_t N = (size_t) 1024*1024*1024/sizeof(double);
  double GB = (double) N*sizeof(double) / (1024.0*1024.0*1024.0);
  omp_set_default_device(0);
  //X = (double*)   omp_target_alloc(N*sizeof(double), omp_get_default_device()) ;
  //Y = (double*)   omp_target_alloc(N*sizeof(double), omp_get_default_device()) ;
  //Z = (double*)   omp_target_alloc(N*sizeof(double), omp_get_default_device()) ;
  //X = new double[N]; //as bad as 16 byte alignment in posix_memalign
  

  
  #ifdef USE_HIP_MALLOC
  hipMalloc( (void**) &X, N*sizeof(double) ); 
  hipMalloc( (void**) &Y, N*sizeof(double) );
  hipMalloc( (void**) &Z, N*sizeof(double) );
  #elif USE_POSSIX_MEMALIGN
  posix_memalign((void**) &X, 512, sizeof(double)*N);
  posix_memalign((void**) &Y, 512, sizeof(double)*N);
  posix_memalign((void**) &Z, 512, sizeof(double)*N);
  #else
  X = new double[N]; //as bad as 16 byte alignment in posix_memalign
  Y = new double[N]; //as bad as 16 byte alignment in posix_memalign
  Z = new double[N]; //as bad as 16 byte alignment in posix_memalign
  #endif


  //#pragma omp target data enter map(alloc:X[0:N], Y[0:N], Z[0:N]) //test and remove is_device_ptr

  const int nthreads = 1024; //if not const - performance decreases (  __launch_bound__ issue ?)
  int nteams = 440*(1024/nthreads)/2;
  //int nteams = (N+nthreads-1)/nthreads;
  int iter  = 5;
  double t1,t2;

 #if 1 //if set to "0"   perfoamnce with "no wait" drops a lot !
  //fill data on CPU
  for (int i = 0; i < iter; ++i){
    t1=omp_get_wtime();
  
    #pragma omp parallel for 
    for (size_t i = 0; i < N; ++i){
       X[i] = 0.000001*i;
       Y[i] = 0.000003*i;
       Z[i] = 0.000005*i;
    }
    t2=omp_get_wtime();
    printf("CPU memory initialization:  loop time  = %g BW = %g[GB/s]\n",(t2-t1), 3.0*GB/(t2-t1) );
  }

#endif


  #pragma omp target  enter data  map(to:X[0:N])
  #pragma omp target  enter data  map(to:Y[0:N])

  for (int cycle = 0; cycle < 4; ++cycle){
 
  //warm up 
  for (int i = 0; i < iter; ++i){
    t1=omp_get_wtime();
    //if (cycle%2 == 0) nteams = 1;
    //else nteams = 440*(1024/nthreads)/2;
    //printf("A:  omp_get_num_threads() = %d omp_get_num_teams()=%d \n",omp_get_num_threads(), omp_get_num_teams());
    
    //nteams*thread_limit <= OMP_NUM_THREADS

//    #pragma omp target teams distribute parallel for num_teams((cycle%2==0)?1:nteams) thread_limit(nthreads) if(target:cycle%2)   //is_device_ptr(X,Y) 
    #pragma omp target teams distribute parallel for num_teams(nteams) thread_limit(nthreads) is_device_ptr(X,Y) 

    for (size_t i = 0; i < N; ++i){
       Y[i] = X[i];
       //if (i == 0) printf("B: omp_get_num_threads() = %d omp_get_num_teams()=%d \n",omp_get_num_threads(), omp_get_num_teams());
    }
    t2=omp_get_wtime();
    printf("GPU:  loop time  = %g BW = %g[GB/s]\n",(t2-t1), 2.0*GB/(t2-t1) );
  }

   for (int i = 0; i < iter; ++i){
    t1=omp_get_wtime();
    #pragma omp parallel for  
    for (size_t i = 0; i < N; ++i)
       Y[i] = X[i];
    t2=omp_get_wtime();
    printf("CPU:  loop time  = %g BW = %g[GB/s]\n",(t2-t1), 2.0*GB/(t2-t1) );
  }

  }

  for (size_t i = N-2; i < N; ++i)
    printf("Y[%zu] = %g\n",i,Y[i]);

#ifdef USE_HIP_MALLOC
hipFree(X);  hipFree(Y);  hipFree(Z);
#elif USE_POSSIX_MEMALIGN
free(X);  free(Y);  free(Z);
#else
delete[] X;  delete[] Y;  delete[] Z;
#endif  

  return 0;
}

#ifdef USE_MPI
#include <mpi.h>
#endif

#include <omp.h>
#include <stdio.h>
#include <unistd.h>
#include <new>
#include <math.h>

#include <sched.h>

#ifdef USE_ROCTX
#include <roctx.h>
#endif

#pragma omp requires unified_shared_memory

void * provide_umpire_pool(size_t N);
void free_umpire_pool( void * data);


#ifndef USE_DEVICE_ALWAYS
#define USE_DEVICE_ALWAYS 0
#endif

void initialize_data(size_t N, double *data){
     int dev = rand() % 100 + 1;
     #pragma omp target teams distribute parallel for if(target:( (dev%2)==0 || USE_DEVICE_ALWAYS==1 ) )
     for (size_t i = 0; i < N; ++i)  data[i] = 0.001*i;

     if ( (dev%2) == 0 || USE_DEVICE_ALWAYS==1 ) fprintf(stderr,"N = %zu, init on GPU\n",N);
     else                fprintf(stderr,"N = %zu, init on CPU\n",N);

}


void test_data(size_t N, double *data){
     int dev = rand() % 111 + 1;
     #pragma omp target teams distribute parallel for if(target:( (dev%2)==0 || USE_DEVICE_ALWAYS==1 ) )
     for (size_t i = 0; i < N; ++i)  {	     
	if (data[i] != 0.001*i)  printf("error : data[%zu] = %g, expected %g, &data[%zu] = %p\n",i,data[i], 0.001*i, i, &data[i]);
     }

     if ( (dev%2) == 0 || USE_DEVICE_ALWAYS==1 ) fprintf(stderr,"N = %zu, tested on GPU\n",N);
     else                fprintf(stderr,"N = %zu, tested on CPU\n",N);

}


double sum_data(size_t N, double *data){
     int dev = rand() % 150 + 1;
     double sum;
     sum  = 0.0;

#if 0
     #pragma omp target teams distribute parallel for reduction(+:sum)  if(target:( (dev%2)==0 || USE_DEVICE_ALWAYS==1 ))    map(tofrom:sum)  
     for (size_t i = 0; i < N; ++i)  sum += data[i];
#else
     double s_sum[1];
     s_sum[0] = 0.0;
     #pragma omp target teams distribute parallel for   if(target:( (dev%2)==0 || USE_DEVICE_ALWAYS==1 ))   //map(tofrom:sum) 
     for (size_t i = 0; i < N; ++i) {
        #pragma omp atomic 
	     s_sum[0] += data[i];
     }
     sum = s_sum[0];

#endif

     if ( (dev%2) == 0 || USE_DEVICE_ALWAYS==1 ) fprintf(stderr,"N = %zu, sum on GPU\n",N);
     else                fprintf(stderr,"N = %zu, sum on CPU\n",N);

     //double s = sum[0]; 
     return sum;
}



int main(int argc, char **argv){

     #ifdef USE_MPI
     MPI_Init(&argc, &argv);
     #endif

    double max_rel_error  = 0;
    size_t Nmax = 0;


    int *cpu_id = new int[omp_get_max_threads()];
    for (int i = 0; i < omp_get_max_threads(); ++i) cpu_id[i] = -1;
    #pragma omp parallel
    {
      cpu_id[omp_get_thread_num()] = sched_getcpu();
    } 
    for (int i = 0; i < omp_get_max_threads(); ++i)
	    fprintf(stderr,"tid = %d; cpuid = %d\n",i,cpu_id[i]);

    delete[] cpu_id;


    for (int o_iter = 1; o_iter < 10000; o_iter+=1){

      
      size_t N = rand() % (1024*1024*10) +  rand() % (7777) + 8;

      Nmax = Nmax > N ? Nmax : N;

      //size_t N = 9942823;
      double * data;
      #if defined(USE_MEM_POOL)
      data = new (  provide_umpire_pool(N*sizeof(double)) ) double[N]  ;
      #elif defined(USE_DEVICE_ALLOC)
      data = (double*)  omp_target_alloc(N*sizeof(double), omp_get_default_device() ); 
      #else
      data =  new (std::align_val_t(256))   double[N];
      #endif

      double GB = (double) N*sizeof(double) / (1024.0*1024.0*1024.0);

      double result = 0.0;

      initialize_data(N,data);
      test_data(N,data);


      result = sum_data(N,data);

      double expected_result = double (N) / 2.0 * (N-1) * 0.001;
      double rel_error = (result-expected_result)/expected_result;
      
      fprintf(stderr," expected CPU result = %g, computed = %g , relative error = %e \n", expected_result, result, rel_error );
      max_rel_error = max_rel_error > fabs(rel_error) ? max_rel_error : fabs(rel_error);


      if (fabs(rel_error) > 1e-12){
	    fprintf(stderr,"STOPPING due to error N = %zu expected CPU result = %e computed = %e relative error = %e\n",N, expected_result, result, rel_error);
	    break;
      }


      double t1 = omp_get_wtime();
      #if defined(USE_MEM_POOL)
      free_umpire_pool( (void*) data );
      #elif defined(USE_DEVICE_ALLOC)
      omp_target_free(data,omp_get_default_device());
      #else
      delete[] data;
      #endif
      double t2 = omp_get_wtime();
      fprintf(stderr, "o_iter = %d  delete : elapsed time = %g [s], effective BW = %g [GB/s] , array size = %g [GB]\n", o_iter, t2-t1, GB/(t2-t1), GB);

    }

    fprintf(stderr,"FINALIZING  max_rel_error = %g Nmax = %zu \n",max_rel_error, Nmax);

    #ifdef USE_MPI
    MPI_Finalize();
    #endif

    return 0;

}

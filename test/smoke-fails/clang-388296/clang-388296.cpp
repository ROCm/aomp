#include <stdio.h>
#include <omp.h>

#pragma omp requires unified_shared_memory 

int main(){

  int N = 2024;
  int *int_array;
  double *dbl_array;


  int_array = new int[N];
  dbl_array = new double[N];

  #pragma omp target teams distribute parallel for
  for (int i = 0; i < N; ++i){
     int_array[i] = i;
     dbl_array[i] = 0.01*i;
  }

#ifdef USE_UNIFIED_MEMORY_MINMAX
  int * min = new int[1];
  int * max = new int[1];
  double *dbl_min = new double[1];
  double *dbl_max = new double[1];
#else
  int *min = (int*) omp_target_alloc(sizeof(int)*1, omp_get_default_device() );
  int *max = (int*) omp_target_alloc(sizeof(int)*1, omp_get_default_device() );
  double *dbl_min = (double*) omp_target_alloc(sizeof(double)*1, omp_get_default_device() );
  double *dbl_max = (double*) omp_target_alloc(sizeof(double)*1, omp_get_default_device() );
#endif

  min[0] = 999998;
  max[0] = 0;
  dbl_min[0] = 999998;
  dbl_max[0] = 0;
  #pragma omp target teams distribute parallel for  //map(tofrom:min[0:1],max[0:1])
  for (int i = 0; i < N; ++i){

    #ifdef USE_UNIFIED_MEMORY_MINMAX
       #pragma omp atomic compare 
       dbl_min[0] = dbl_array[i] < dbl_min[0] ? dbl_array[i] : dbl_min[0]; // MIN

       #pragma omp atomic compare 
       dbl_max[0] = dbl_array[i] > dbl_max[0] ? dbl_array[i] : dbl_max[0]; //MAX 

    #else	  
       #pragma omp atomic compare hint(ompx_fast_fp_atomics)
       dbl_min[0] = dbl_array[i] < dbl_min[0] ? dbl_array[i] : dbl_min[0]; // MIN

       #pragma omp atomic compare hint(ompx_fast_fp_atomics)
       dbl_max[0] = dbl_array[i] > dbl_max[0] ? dbl_array[i] : dbl_max[0]; //MAX   
    #endif
  }

  fprintf(stderr,"dbl_max[0] = %g, dbl_min[0] = %g\n",dbl_max[0],dbl_min[0]);


  int min2 = 999992;
  int max2 = 0;
  #pragma omp target teams distribute parallel for  map(tofrom:min2,max2)
  for (int i = 0; i < N; ++i){

    #pragma omp atomic compare  //hint(ompx_fast_fp_atomics)
    min2 = int_array[i] < min2 ? int_array[i] : min2; // MIN

    #pragma omp atomic compare //hint(ompx_fast_fp_atomics)
    max2 = int_array[i] > max2 ? int_array[i] : max2; //MAX   
  }

  fprintf(stderr,"max2 = %d, min2 = %d\n",max2,min2);



  return 0;
}

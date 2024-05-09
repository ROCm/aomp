#include <stdio.h>
#include <omp.h>
//%env: LIBOMPTARGET_KERNEL_TRACE=1
int main(void) {
   #pragma omp target 
      #pragma omp parallel // num_threads(128)
         printf("Hello4. Hello GPU world, THREAD %d of %d \n", 
                omp_get_thread_num(), omp_get_num_threads());
}

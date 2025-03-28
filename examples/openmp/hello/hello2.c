#include <stdio.h>
#include <omp.h>
int main(void) {
   #pragma omp parallel // num_threads(8)
      printf("Hello2. Hello CPU world, THREAD %d of %d \n", 
             omp_get_thread_num(), omp_get_num_threads());
}

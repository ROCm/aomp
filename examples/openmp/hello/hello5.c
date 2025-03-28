#include <stdio.h>
#include <omp.h>
///%cflags: -v
//%env: LIBOMPTARGET_KERNEL_TRACE=1
int main(void) {
#pragma omp target teams distribute parallel for // thread_limit(512)
// #pragma omp parallel for
    for(int i = 0; i < 4096; i++){
      if((omp_get_thread_num() == 0) || (i == 4000))  // this if statement reduces output
         printf("Hello5. %d Hello GPU world.  THREAD %d of %d   TEAM %3d of %3d\n",
                i, omp_get_thread_num(), omp_get_num_threads(), 
                omp_get_team_num(), omp_get_num_teams());
    }
}

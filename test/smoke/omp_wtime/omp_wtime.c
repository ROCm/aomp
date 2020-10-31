#include <omp.h>
#include <stdio.h>

int main()
{
  double time0, time1;
#pragma omp target map(from: time0, time1)
  {
    // calls to wtime should not be folded together
   time0 = omp_get_wtime();
#if defined(__AMDGCN__)
   // nvptx has a nanosleep, but only for >= sm_70
   __builtin_amdgcn_s_sleep(1000);
#endif
   time1 = omp_get_wtime();
  }

   double delta = time1 - time0;
   if(delta)
     printf("Success!\n");
   else
     printf("Failure\n");
   return delta == 0.; // success if they differ
}

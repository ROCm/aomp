#include <stdio.h>
#include <omp.h>
/*
 unify effect of nest-var ICV and max-active-levels-var ICV on nested parallelism; deprecate OMP_NESTED, omp_get_nested, and omp_set_nested and add omp_get_supported_active_levels
 */
int main(){
   int n = omp_get_nested();
   omp_set_nested(1);
   fprintf(stderr,"getnest = %d\n",n);
   n = omp_get_supported_active_levels();
   fprintf(stderr,"get_supported_levels = %d\n",n);
  return 0;
}

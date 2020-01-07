#include <stdio.h>
#define N   10


int main (void)
{
  long int aa=0;
  int res = 0;

  int ng =12;
  int cmom = 14;
  int nxyz = 5000;
// fails for 149 and above: nxyz=149;
// for testing: nxyz = 10;
  #pragma omp target teams distribute num_teams(nxyz) thread_limit(ng*(cmom-1)) map(tofrom:aa)
  for (int gid = 0; gid < nxyz; gid++) {
    #pragma omp parallel for  collapse(2) 
    for (unsigned int g = 0; g < ng; g++) {
      for (unsigned int l = 0; l < cmom-1; l++) {
        int a = 0;
        #pragma omp parallel for reduction(+:a)
        for (int i = 0; i < N; i++) {
          a += i;
        }
        #pragma omp atomic 
        aa += a;
      }
    }  
  }
  long exp = (long)ng*(cmom-1)*nxyz*(N*(N-1)/2);
  printf ("The result is = %ld exp:%ld!\n", aa,exp);
  if (aa != exp) {
    printf("Failed %ld\n",aa);
    return 1;
  }
  return 0;
}

#include <stdio.h>
#define N   10


int main (void)
{
  long int a=0;
  int res = 0;

  int ng =12;
  int cmom = 14;
  int nxyz = 50;
// fails for 149 and above: nxyz=149;
// for testing: nxyz = 10;
  #pragma omp target teams distribute  map(tofrom:a)
  for (int gid = 0; gid < nxyz; gid++) {
    #pragma omp parallel for  collapse(2) 
    for (unsigned int g = 0; g < ng; g++) {
      for (unsigned int l = 0; l < cmom-1; l++) {
        for (int i = 0; i < N; i++) {
          a += i;
        }
      }
    }  
  }
  printf ("The result is = %d\n", a);
  return 0;
}

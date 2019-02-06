#include <stdio.h>

/*  This demo shows that reduction is not working for nvidia */

int main (void)
{
  int SUM1=0,SUM2=0;
  int N1=32,N2=64;

  #pragma omp target map(tofrom:SUM1) 
  #pragma omp teams distribute parallel for reduction(+:SUM1) 
  for (int i = 1; i <= N1 ; i++) 
    SUM1 += i;

  #pragma omp target map(tofrom:SUM2) 
  #pragma omp teams distribute parallel for reduction(+:SUM2) 
  for (int i = 1; i <= N2 ; i++) 
    SUM2 += i;

  printf("N1:%d  Sum1:%d  CheckVal:%d\n",N1,SUM1,(N1*(N1+1))/2);
  printf("N2:%d  Sum2:%d  CheckVal:%d\n",N2,SUM2,(N2*(N2+1))/2);

  if ( (SUM1 != (N1*(N1+1))/2)  ||
       (SUM2 != (N2*(N2+1))/2) ) { 
    printf("INVALID RESULTS\n");
    return 1;
  } else  {
    printf("Success\n");
    return 0;
  } 
}

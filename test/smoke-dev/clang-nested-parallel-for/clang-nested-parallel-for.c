#include <stdio.h>

int main () {
  int a[1024];
  int i,j;

#pragma omp target teams distribute parallel for map(tofrom:a)
 for (i =0; i <16; i++)
#pragma omp parallel for
   for (j=0; j< 64; j++)
     a[i*64 + j] = (i*64 + j);

  for (i=0; i<1024;i++)
     if (a[i] != i)
       return 1;
  return 0;
}

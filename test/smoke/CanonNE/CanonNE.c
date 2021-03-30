#include <stdio.h>
#include <omp.h>
int main()
{
  int numTeams=12800;
  int foo = 0;
#pragma omp target teams distribute parallel for 
  for (int j=0; j != numTeams; j++) {
     foo++;
  }
  printf("%d\n",foo);
return 0;
}



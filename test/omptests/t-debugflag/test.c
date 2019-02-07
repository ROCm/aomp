#include <omp.h>
#include <stdio.h>

int main()
{
  int Success = 0;

  #pragma omp target map(from:Success)
  {
      Success = 1;
  }

  if (Success)
    printf("### Success ### \n");
  else
    printf("### Error ###\n"); 
  return 0;
}


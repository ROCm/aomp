#include <omp.h>
#include <stdio.h>


int main (){

#define N 1024
  double x_d[N];
  for (size_t i = 0; i < N; ++i)
    x_d[i] = -1;
  printf("x_d = %p\n",x_d);

  #pragma omp target teams
  {
    int lower = 0;
  #pragma omp distribute parallel for 
    for (size_t i = lower; i < N; ++i)
       x_d[i] = i;
  }
  printf("x_d[1] = %f\n", x_d[1]);
  return 0;
}

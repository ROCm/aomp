#include "omp.h"
#include <stdio.h>

int main(int argc, char* argv[])
{
  int t_size = 1;
  int x_size = 4;
  int y_size = 4;
  int z_size = 4;

  int time_m = 0, time_M = t_size - 1;
  int x_m = 0, x_M = x_size - 1;
  int y_m = 0, y_M = y_size - 1;
  int z_m = 0, z_M = z_size - 1;

  int *u_vec2 = (int*) malloc(sizeof(int)*t_size*x_size*y_size*z_size);
  int (*restrict u)[x_size][y_size][z_size] 
#define USE_ALIGNED
#ifdef USE_ALIGNED
	 __attribute__ ((aligned (64)))
#endif
	 = (int (*)[x_size][y_size][z_size]) u_vec2;

  for (int t = time_m; t <= time_M; t++)
    for (int x = x_m; x <= x_M; x++)
      for (int y = y_m; y <= y_M; y++)
        for (int z = z_m; z <= z_M; z++)
          u[t][x][y][z] = 0;

  #pragma omp target enter data map(to: u[0:t_size][0:x_size][0:y_size][0:z_size])

  for (int t = time_m; t <= time_M; t++)
  {
    #pragma omp target teams distribute parallel for collapse(3)
    for (int x = x_m; x <= x_M; x++)
    {
      for (int y = y_m; y <= y_M; y++)
      {
        for (int z = z_m; z <= z_M; z++)
        {
          u[t][x][y][z] = u[t][x][y][z] + 1;
        }
      }
    }
  }

  #pragma omp target update from(u[0:t_size][0:x_size][0:y_size][0:z_size])
  #pragma omp target exit data map(release: u[0:t_size][0:x_size][0:y_size][0:z_size])

  for (int t = time_m; t <= time_M; t++)
    for (int x = x_m; x <= x_M; x++)
      for (int y = y_m; y <= y_M; y++)
        for (int z = z_m; z <= z_M; z++)
          printf("%d ", u[t][x][y][z]);
  printf("\n");

  return 0;
}


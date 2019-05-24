#include <stdio.h>
#include <omp.h>
int main()
{
  int N = 1024;
  int NN = N*N;
  int Res[NN];
  for (int i=0; i < NN; i++) Res[i] = -1;

  #pragma omp target teams thread_limit(1024) num_teams(N)
  #pragma omp distribute parallel for
  for (int j=0; j < NN; j++) { 
    if (j==12) printf("teams %d threads %d\n",omp_get_num_teams(), omp_get_num_threads());
    Res[j] = j;
  }
  for (int i=0; i < NN; i++)
    if (Res[i] != i) {
      printf("Failed %d %d\n",i, Res[i]);
      return 1;
    }
  return 0;
}


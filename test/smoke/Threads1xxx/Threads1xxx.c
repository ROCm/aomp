#include <stdio.h>
#include <omp.h>
int thdLim =1024;
//__attribute__((amdgpu_flat_work_group_size(1024, 1024))) 
int main()
{
  int N = 12;
  int NN = 1024;
  int lims[N] , threads[N], Res[N*NN];
  int i;
  for (i=0; i <N; i++) lims[i] = threads[i] = -1;
  for (i=0; i <N*NN; i++) Res[i] = -1;
  int numTeams=64;
#pragma omp target teams num_teams(numTeams) thread_limit(1024)
#pragma omp parallel for
  for (i=0; i <NN*numTeams; i++) {
    if (i<N) {
      lims[i%N] = omp_get_num_teams();
      threads[i%N] = omp_get_num_threads();
    }
    Res[i] = i;
  }
  for (i=0; i <N*NN; i++) if (Res[i] != i) { printf("Failed\n"); break;}
  for (i=0; i <N; i++) {
    printf("i=%d lims[%d] threads[%d]\n", i, lims[i], threads[i]);
  }
return 0;
}



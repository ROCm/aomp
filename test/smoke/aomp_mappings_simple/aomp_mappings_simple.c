#include <stdio.h>
#include <omp.h>
int main()
{
  int N = 128;
  int NN = 1024;

  int team_num[NN];
  int smid[NN];
  int i;

  for (i=0; i<NN; i++)
   team_num[i]=smid[i]= -1;

#pragma omp target teams
  {    
       int j = omp_get_team_num();
       team_num[j] = omp_get_team_num();
       smid[j] = omp_ext_get_smid();
  }

  int rc = 0;
  fprintf(stderr,"    i team# smid\n");
  for (i=0; i<N; i++)
    fprintf(stderr," %4d  %4d  %4d\n",
    i,team_num[i],smid[i]);
  return rc;
}



#include <stdio.h>
#include <omp.h>

int main (void)
{
  int num_teams = 0;
  #pragma omp target teams distribute num_teams(2) map(tofrom: num_teams)
  for(int j = 0; j < omp_get_num_teams(); j++)
  {
    num_teams = omp_get_num_teams();
    printf ("The number of teams = %d!\n", omp_get_num_teams());
  }
  if(num_teams != 2){
    printf("FAILURE! Number of teams is %d and should be 2.\n", num_teams);
    return 1;
  }
  printf("SUCCESS!\n");
  return 0;
}

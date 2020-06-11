#include <stdio.h>
#include <omp.h>



int main(void) {

  int fail = 0;

  //
  // Test: num_teams and omp_get_team_num()
  #pragma omp target
 {
   printf("Num_teams=%d\n", omp_get_num_teams());
 }
  #pragma omp target
 {
    #pragma omp teams
    {
      if (omp_get_team_num() == 0)
        printf("Num_teams=%d\n", omp_get_num_teams());
      #pragma omp distribute
      for (int i=0; i< 10; i++)
        printf("team %d thread %d\n", omp_get_team_num(), omp_get_thread_num());
    }
  }

  return fail;
}


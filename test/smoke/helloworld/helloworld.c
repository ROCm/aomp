#include <stdio.h>
#include <omp.h>

#define  GET_THREAD_NUM 0
#define  GET_NUM_THREADS 1
#define  GET_TEAM_NUM 2
#define  GET_NUM_TEAMS 3
#define  GET_MAX_THREADS 4
#define  GET_LEVEL 5
#define  GET_NUM_PROCS 6
#define  GET_TEAM_SIZE 7
#define  GET_TEAM_SIZE1 8
#define  GET_TEAM_SIZE0 9
#define  LAST_ITER 10

int main(void) {
int ibuf[16];
ibuf[GET_THREAD_NUM] = -1;

for(int j = 0; j < 1; j++) {

#pragma omp target teams distribute parallel for map(from:ibuf[0:16])
{
  for( int i=0 ; i<259 ; i++) {
    ibuf[GET_THREAD_NUM] = omp_get_thread_num();
    ibuf[GET_NUM_THREADS] = omp_get_num_threads();
    ibuf[GET_TEAM_NUM] = omp_get_team_num();
    ibuf[GET_NUM_TEAMS] = omp_get_num_teams();
    ibuf[GET_MAX_THREADS] = omp_get_max_threads();
    ibuf[GET_LEVEL] = omp_is_initial_device();
    ibuf[GET_NUM_PROCS] = omp_get_num_procs();
    // ibuf[GET_TEAM_SIZE] = omp_get_team_size(omp_get_level());
    if(omp_get_thread_num() > 128)
      ibuf[GET_TEAM_SIZE1] = omp_ext_get_warp_id();
    ibuf[GET_TEAM_SIZE0] = omp_ext_is_spmd_mode();
    ibuf[LAST_ITER] = i;
    // printf(" warp id : %d\n", omp_ext_get_warp_id());
  }
}
}
  printf("  omp_get_thread_num() %d\n",ibuf[GET_THREAD_NUM]);
  printf("  omp_get_num_threads() %d\n",ibuf[GET_NUM_THREADS]);
  printf("  omp_get_team_num() %d\n",ibuf[GET_TEAM_NUM]);
  printf("  omp_get_num_teams() %d\n",ibuf[GET_NUM_TEAMS]);
  printf("  omp_get_max_threads() %d\n",ibuf[GET_MAX_THREADS]);
  printf("  omp_is_initial_device() %d\n",ibuf[GET_LEVEL]);
  printf("  omp_get_num_procs() %d\n",ibuf[GET_NUM_PROCS]);
//   printf("  omp_get_team_size(level) %d\n",ibuf[GET_TEAM_SIZE]);
  printf("  omp_ext_get_warp_id() %d\n",ibuf[GET_TEAM_SIZE1]);
  printf("  omp_ext_is_spmd_mode() %d\n",ibuf[GET_TEAM_SIZE0]);
  printf("  last_iter %d\n",ibuf[LAST_ITER]);
  return 0;
}


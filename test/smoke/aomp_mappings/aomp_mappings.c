#include <stdio.h>
#include <omp.h>
int main()
{
  int N = 102;

  int thread_num[N];
  int team_num[N];
  int default_dev[N];
  int warp_id[N];
  int lane_id[N];
  int smid[N];
  int is_spmd_mode[N];
  int master_thread_id[N];
  int num_teams[N];
  int num_threads[N];
  unsigned long long active_mask[N];
  int i;

  for (i=0; i<N; i++)
    active_mask[i] = 0;
  for (i=0; i<N; i++)
    thread_num[i]=team_num[i]=default_dev[i]=warp_id[i]=lane_id[i]=master_thread_id[i]=smid[i]=is_spmd_mode[i] = -1;


fprintf(stderr,"#pragma omp target teams distribute parallel for thread_limit(4)\n");
#pragma omp target teams distribute parallel for thread_limit(4)
  {
    for (int j = 0; j< N; j++) {
       thread_num[j] = omp_get_thread_num();
       num_threads[j] = omp_get_num_threads();
       team_num[j] = omp_get_team_num();
       num_teams[j] = omp_get_num_teams();
       default_dev[j] = omp_get_default_device();
       warp_id[j] = omp_get_warp_id();
       lane_id[j] = omp_get_lane_id();
       active_mask[j] = omp_get_active_threads_mask();
       smid[j] = omp_get_smid();
       master_thread_id[j] = omp_get_master_thread_id();
       is_spmd_mode[j] = omp_is_spmd_mode();
    }
  }
  fprintf(stderr,"    i thrd# team#  dev# warp# lane# MastThrd smid  SPMD num_threads num_teams ActiveMask\n");
  for (i=0; i<N; i++)
    fprintf(stderr," %4d  %4d  %4d  %4d  %4d  %4d  %4d  %4d  %4d %10d %10d %16llx\n",
    i,thread_num[i],team_num[i],default_dev[i],warp_id[i],lane_id[i],master_thread_id[i],smid[i],is_spmd_mode[i],num_threads[i], num_teams[i],active_mask[i]);

fprintf(stderr,"#pragma omp target teams distribute parallel for\n");
#pragma omp target teams distribute parallel for
  {
    for (int j = 0; j< N; j++) {
       thread_num[j] = omp_get_thread_num();
       num_threads[j] = omp_get_num_threads();
       team_num[j] = omp_get_team_num();
       num_teams[j] = omp_get_num_teams();
       default_dev[j] = omp_get_default_device();
       warp_id[j] = omp_get_warp_id();
       lane_id[j] = omp_get_lane_id();
       active_mask[j] = omp_get_active_threads_mask();
       smid[j] = omp_get_smid();
       master_thread_id[j] = omp_get_master_thread_id();
       is_spmd_mode[j] = omp_is_spmd_mode();
    }
  }
  fprintf(stderr,"    i thrd# team#  dev# warp# lane# MastThrd smid  SPMD num_threads num_teams ActiveMask\n");
  for (i=0; i<N; i++)
    fprintf(stderr," %4d  %4d  %4d  %4d  %4d  %4d  %4d  %4d  %4d %10d %10d %16llx\n",
    i,thread_num[i],team_num[i],default_dev[i],warp_id[i],lane_id[i],master_thread_id[i],smid[i],is_spmd_mode[i],num_threads[i], num_teams[i],active_mask[i]);

fprintf(stderr,"#pragma omp target teams \n");
#pragma omp target teams
//#pragma omp parallel
  {
    for (int j = 0; j< N; j++) {
       thread_num[j] = omp_get_thread_num();
       num_threads[j] = omp_get_num_threads();
       team_num[j] = omp_get_team_num();
       num_teams[j] = omp_get_num_teams();
       default_dev[j] = omp_get_default_device();
       warp_id[j] = omp_get_warp_id();
       lane_id[j] = omp_get_lane_id();
       active_mask[j] = omp_get_active_threads_mask();
       smid[j] = omp_get_smid();
       master_thread_id[j] = omp_get_master_thread_id();
       is_spmd_mode[j] = omp_is_spmd_mode();
    }
  }

  int rc = 0;
  fprintf(stderr,"    i thrd# team#  dev# warp# lane# MastThrd smid  SPMD num_threads num_teams ActiveMask\n");
  for (i=0; i<N; i++)
    fprintf(stderr," %4d  %4d  %4d  %4d  %4d  %4d  %4d  %4d  %4d %10d %10d  %16llx\n",
    i,thread_num[i],team_num[i],default_dev[i],warp_id[i],lane_id[i],master_thread_id[i],smid[i],is_spmd_mode[i],num_threads[i],num_teams[i],active_mask[i]);
  return rc;
}



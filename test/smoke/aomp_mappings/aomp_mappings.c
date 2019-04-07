#include <stdio.h>
#include <omp.h>
int main()
{
  int N = 128;
  int NN = 1024;

  int thread_num[NN];
  int team_num[NN];
  int default_dev[NN];
  int warp_id[NN];
  int lane_id[NN];
  int smid[NN];
  int is_spmd_mode[NN];
  int master_thread_id[NN];
  int num_teams[NN];
  int num_threads[NN];
  unsigned long long active_mask[NN];
  int i;

  for (i=0; i<NN; i++)
    active_mask[i] = 0;
  for (i=0; i<NN; i++)
    thread_num[i]=team_num[i]=default_dev[i]=warp_id[i]=lane_id[i]=master_thread_id[i]=smid[i]=is_spmd_mode[i]=num_threads[i]=num_teams[i] = -1;


fprintf(stderr,"#pragma omp target teams distribute parallel for thread_limit(4)\n");
#pragma omp target teams distribute parallel for thread_limit(4)
  {
    for (int j = 0; j< N; j++) {
       thread_num[j] = omp_get_thread_num();
       num_threads[j] = omp_get_num_threads();
       team_num[j] = omp_get_team_num();
       num_teams[j] = omp_get_num_teams();
       default_dev[j] = omp_get_default_device();
       warp_id[j] = omp_ext_get_warp_id();
       lane_id[j] = omp_ext_get_lane_id();
       active_mask[j] = omp_ext_get_active_threads_mask();
       smid[j] = omp_ext_get_smid();
       master_thread_id[j] = omp_ext_get_master_thread_id();
       is_spmd_mode[j] = omp_ext_is_spmd_mode();
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
       warp_id[j] = omp_ext_get_warp_id();
       lane_id[j] = omp_ext_get_lane_id();
       active_mask[j] = omp_ext_get_active_threads_mask();
       smid[j] = omp_ext_get_smid();
       master_thread_id[j] = omp_ext_get_master_thread_id();
       is_spmd_mode[j] = omp_ext_is_spmd_mode();
    }
  }
  fprintf(stderr,"    i thrd# team#  dev# warp# lane# MastThrd smid  SPMD num_threads num_teams ActiveMask\n");
  for (i=0; i<N; i++)
    fprintf(stderr," %4d  %4d  %4d  %4d  %4d  %4d  %4d  %4d  %4d %10d %10d %16llx\n",
    i,thread_num[i],team_num[i],default_dev[i],warp_id[i],lane_id[i],master_thread_id[i],smid[i],is_spmd_mode[i],num_threads[i], num_teams[i],active_mask[i]);

fprintf(stderr,"#pragma omp target teams \n");
#pragma omp target teams
  {
       int j = omp_get_team_num();
       thread_num[j] = omp_get_thread_num();
       num_threads[j] = omp_get_num_threads();
       team_num[j] = omp_get_team_num();
       num_teams[j] = omp_get_num_teams();
       default_dev[j] = omp_get_default_device();
       warp_id[j] = omp_ext_get_warp_id();
       lane_id[j] = omp_ext_get_lane_id();
       active_mask[j] = omp_ext_get_active_threads_mask();
       smid[j] = omp_ext_get_smid();
       master_thread_id[j] = omp_ext_get_master_thread_id();
       is_spmd_mode[j] = omp_ext_is_spmd_mode();
  }

  int rc = 0;
  fprintf(stderr,"    i thrd# team#  dev# warp# lane# MastThrd smid  SPMD num_threads num_teams ActiveMask\n");
  for (i=0; i<N; i++)
    fprintf(stderr," %4d  %4d  %4d  %4d  %4d  %4d  %4d  %4d  %4d %10d %10d  %16llx\n",
    i,thread_num[i],team_num[i],default_dev[i],warp_id[i],lane_id[i],master_thread_id[i],smid[i],is_spmd_mode[i],num_threads[i],num_teams[i],active_mask[i]);
  return rc;
}



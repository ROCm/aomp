#include <stdio.h>
#include <omp.h>
#include <string.h>

//Shared Variables
int THREAD_LIMIT = 4;
int MAX_TEAMS = 128;
int GENERIC = 0;
int SPMD = 1;
int MAX_THREADS_PER_TEAM = 256;

#ifdef WAVE_SIZE
  int WARP_SIZE = WAVE_SIZE;
#else
  int WARP_SIZE = 64;
#endif
/*
 * Function: recordError
 * Description: Updates error number and prints error messages
 */
void recordError(int* error , char *message, int iteration, int * array, unsigned long long *mask ){
  (*error)++;
  if(mask == NULL)
    fprintf(stderr,"%s IS INCORRECT! Iteration: %d Value: %d\n", message, iteration, array[iteration]);

  else
    fprintf(stderr,"%s IS INCORRECT! Iteration: %d Value: %llx\n", message, iteration, mask[iteration]);
}

int main()
{
  printf("warpsize %d\n", WARP_SIZE);

  //Determine which GPU type (NVIDIA or AMD)
  char* nvidia= "sm";
  char* aomp_gpu= getenv("AOMP_GPU");
  int isAMDGPU = 1;
  if(aomp_gpu && strstr(aomp_gpu, nvidia) != NULL)
    isAMDGPU = 0;

  // a hacky way to know the default number of teams
  #pragma omp target teams map(tofrom:MAX_TEAMS)
  {
    if (omp_get_team_num() == 0)
      MAX_TEAMS = omp_get_num_teams();
  }
  fprintf(stderr, "MAX_TEAMS: %d\n", MAX_TEAMS);
  //Logic for correct shared variables - AMD vs NVIDIA GPU
  if(!isAMDGPU){
    printf("%s\n", getenv("AOMP_GPU"));
    MAX_THREADS_PER_TEAM = 128;
    WARP_SIZE = 32;
  }

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
  unsigned long long mask = 0;
  int i;
  int correctTeamNum = -1;
  int correctNumTeams = -1;
  int correctWarpId = -1;
  int remainder = 0;
  int errors = 0;

  //Initialize arrays
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

  //Verify Results - #pragma omp target teams distribute parallel for thread_limit(4)
  for (i = 0; i < N; i++){

    //check thread #
    if (thread_num[i] != i % THREAD_LIMIT)
      recordError(&errors, "THREAD NUMBER", i, thread_num, NULL);

    //check team #
    if (i % THREAD_LIMIT == 0){
      correctTeamNum++;
      if(isAMDGPU)
        correctTeamNum = correctTeamNum % MAX_TEAMS;
    }
    if (team_num[i] != correctTeamNum)
      recordError(&errors, "TEAM NUMBER", i, team_num, NULL);

    //check device #, We use default device (0) for testing
    if (default_dev[i] != 0)
      recordError(&errors, "DEVICE NUMBER", i, default_dev, NULL);

    //check warp #
    if (warp_id[i] != 0)
      recordError(&errors, "WARP NUMBER", i, warp_id, NULL);

    //check lane #
    if (lane_id[i] != i % THREAD_LIMIT)
      recordError(&errors, "LANE NUMBER", i, lane_id, NULL);

    //check master thread #
    if (master_thread_id[i] != 0 )
      recordError(&errors, "MASTER THREAD NUMBER", i, master_thread_id, NULL);

    //check SPMD mode #
    if (is_spmd_mode[i] != SPMD )
      recordError(&errors, "SPMD NUMBER", i, is_spmd_mode, NULL);

    //check num threads
    if (num_threads[i] != THREAD_LIMIT )
      recordError(&errors, "NUM THREADS", i, num_threads, NULL);

    //check num teams
    //If number of iterations is not divisible by THREAD_LIMIT get the ceiling
    if(N % THREAD_LIMIT != 0)
      correctNumTeams = ((N + num_threads[i]) / num_threads[i]);
    else
      correctNumTeams = N / THREAD_LIMIT;
    if (correctNumTeams > MAX_TEAMS && isAMDGPU)
      correctNumTeams = MAX_TEAMS;

    if (num_teams[i] != correctNumTeams)
      recordError(&errors, "NUM TEAMS", i, num_teams, NULL);

    //check active mask
    mask = 0;
    if(N % THREAD_LIMIT != 0){
      remainder = N % THREAD_LIMIT;

      //set bit mask to proper value
      for (int j = 0 ; j < remainder; j++){
        mask = mask << 1;
        mask = mask + 1;
      }

    }

    //Mask for last evenly divided iteration
    if (i < N - remainder){
      mask = 0xf;
    }
    if (active_mask[i] != mask)
      recordError(&errors, "ACTIVE MASK", i, NULL, active_mask);
  }

  //Reset Arrays
  for (i=0; i<NN; i++)
    active_mask[i] = 0;
  for (i=0; i<NN; i++)
    thread_num[i]=team_num[i]=default_dev[i]=warp_id[i]=lane_id[i]=master_thread_id[i]=smid[i]=is_spmd_mode[i]=num_threads[i]=num_teams[i] = -1;

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

  //Verify Results - #pragma omp target teams distribute parallel for
  correctTeamNum = -1;
  correctNumTeams = -1;
  //int correctWarpId = -1;

  //Verify Results
  for (i = 0; i < N; i++){

    //check thread #
    if (thread_num[i] != i % MAX_THREADS_PER_TEAM)
      recordError(&errors, "THREAD NUMBER", i, thread_num, NULL);

    //check team #
    if (i % MAX_THREADS_PER_TEAM == 0){
      correctTeamNum++;
      correctTeamNum = correctTeamNum % MAX_TEAMS;
    }

    if (team_num[i] != correctTeamNum)
      recordError(&errors, "TEAM NUMBER", i, team_num, NULL);

    //check device #, We use default device (0) for testing
    if (default_dev[i] != 0)
      recordError(&errors, "DEVICE NUMBER", i, default_dev, NULL);

    //check warp #
    if (i % WARP_SIZE == 0){
      correctWarpId++;
      correctWarpId = correctWarpId % (MAX_THREADS_PER_TEAM/WARP_SIZE);
    }
    if (warp_id[i] != correctWarpId)
      recordError(&errors, "WARP NUMBER", i, warp_id, NULL);

    //check lane #
    if (lane_id[i] != i % WARP_SIZE)
      recordError(&errors, "LANE NUMBER", i, lane_id, NULL);

    //check master thread #
    if (master_thread_id[i] != MAX_THREADS_PER_TEAM - WARP_SIZE)
      recordError(&errors, "MASTER THREAD NUMBER", i, master_thread_id, NULL);

    //check SPMD mode #
    if (is_spmd_mode[i] != SPMD )
      recordError(&errors, "SPMD NUMBER", i, is_spmd_mode, NULL);

    //check num threads
    if (num_threads[i] != MAX_THREADS_PER_TEAM )
      recordError(&errors, "NUM THREADS", i, num_threads, NULL);

    //check num teams
    //If number of iterations is not divisible by MAX_THREADS_PER_TEAM get the ceiling
    if(N % MAX_THREADS_PER_TEAM != 0)
      correctNumTeams = ((N + num_threads[i]) / num_threads[i]);
    else
      correctNumTeams = N / MAX_THREADS_PER_TEAM;
    if (num_teams[i] != correctNumTeams)
      recordError(&errors, "NUM TEAMS", i, num_teams, NULL);

    //check active mask
    remainder = 0;
    mask = 0;

    //Set mask for 64 or fewer active threads in first warp
    if (N < WARP_SIZE + 1){
      remainder = N % WARP_SIZE;
    }
    else
      remainder = (N % MAX_THREADS_PER_TEAM) % WARP_SIZE;

    //Set mask for warps with full (64) active threads
    if (i < N - remainder){
      if(WARP_SIZE == 64)
        mask = 0xffffffffffffffff;
      else
        mask = 0xffffffff;
    }
    else{ //set mask for iterations with non full warps
      mask = 0;
      for (int j = 0 ; j < remainder; j++){
        mask = mask << 1;
        mask = mask + 1;
      }
    }

    if (active_mask[i] != mask){
      recordError(&errors, "ACTIVE MASK", i, NULL, active_mask);
    }
  }

  //Reset Arrays
  for (i=0; i<NN; i++)
    active_mask[i] = 0;
  for (i=0; i<NN; i++)
    thread_num[i]=team_num[i]=default_dev[i]=warp_id[i]=lane_id[i]=master_thread_id[i]=smid[i]=is_spmd_mode[i]=num_threads[i]=num_teams[i] = -1;

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

  fprintf(stderr,"    i thrd# team#  dev# warp# lane# MastThrd smid  SPMD num_threads num_teams ActiveMask\n");
  for (i=0; i<N; i++)
    fprintf(stderr," %4d  %4d  %4d  %4d  %4d  %4d  %4d  %4d  %4d %10d %10d  %16llx\n",
    i,thread_num[i],team_num[i],default_dev[i],warp_id[i],lane_id[i],master_thread_id[i],smid[i],is_spmd_mode[i],num_threads[i],num_teams[i],active_mask[i]);

//Verify Results - #pragma omp target teams
  correctTeamNum = -1;
  correctNumTeams = -1;

  //Verify Results
  for (i = 0; i < N; i++){
    //Only check iterations up to MAX_TEAMS
    if(i < MAX_TEAMS){
      //check thread #
      if (thread_num[i] != 0)
        recordError(&errors, "THREAD NUMBER", i, thread_num, NULL);

      //check team #
      if (team_num[i] != i)
        recordError(&errors, "TEAM NUMBER", i, team_num, NULL);

      //check device #, We use default device (0) for testing
      if (default_dev[i] != 0)
        recordError(&errors, "DEVICE NUMBER", i, default_dev, NULL);

      //check warp #
      if ((warp_id[i] != (MAX_THREADS_PER_TEAM - WARP_SIZE) / WARP_SIZE) &&
          (warp_id[i] != (MAX_THREADS_PER_TEAM - WARP_SIZE) / WARP_SIZE +1))
        recordError(&errors, "WARP NUMBER", i, warp_id, NULL);

      //check lane #
      if (lane_id[i] != 0)
        recordError(&errors, "LANE NUMBER", i, lane_id, NULL);

      //check master thread #
      if ((master_thread_id[i] != MAX_THREADS_PER_TEAM - WARP_SIZE) &&
          (master_thread_id[i] != MAX_THREADS_PER_TEAM))
        recordError(&errors, "MASTER THREAD NUMBER", i, master_thread_id, NULL);

      //check SPMD mode #
      if (is_spmd_mode[i] != GENERIC )
        recordError(&errors, "SPMD NUMBER", i, is_spmd_mode, NULL);

      //check num threads
      if (num_threads[i] != 1 )
        recordError(&errors, "NUM THREADS", i, num_threads, NULL);

      //check num teams
      //If number of iterations is not divisible by MAX_THREADS_PER_TEAM get the ceiling

      if (num_teams[i] != MAX_TEAMS )
        recordError(&errors, "NUM TEAMS", i, num_teams, NULL);

      //check active mask
      remainder = 0;
      mask = 1;

      if (active_mask[i] != mask){
        recordError(&errors, "ACTIVE MASK", i, NULL, active_mask);
      }
    }
    else{
      if(thread_num[i] != -1 || team_num[i] != -1 || default_dev[i] != -1 || warp_id[i] != -1 || lane_id[i] != -1 || master_thread_id[i] != -1 || is_spmd_mode[i] != -1 || num_threads[i] != -1 || num_teams[i] != -1 || active_mask[i] != 0){
        fprintf(stderr, "Data after iteration %d is changed and should be untouched!!\n", MAX_TEAMS - 1);
	errors++;
      }
    }
  }

 //Print results and return total errors
  if(!errors){
    fprintf(stderr, "Success\n");
    return 0;
  }
  else {
    fprintf(stderr, "Fail\n");
    fprintf(stderr, "Errors: %d\n", errors);
    return 1;
  }
}

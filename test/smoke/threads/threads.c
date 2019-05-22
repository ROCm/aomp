#include <stdio.h>
#include <omp.h>
#include <string.h>

int main()
{
  //Determine which GPU type (NVIDIA or AMD)
  char* nvidia= "sm";
  char* aomp_gpu= getenv("AOMP_GPU");
  int isAMDGPU = 1;
  int masterWarpThread = -1;

  if(strstr(aomp_gpu, nvidia) != NULL)
    isAMDGPU = 0;

  if(isAMDGPU)
    masterWarpThread = 959;
  else
    masterWarpThread = 991;

  int thread_id[1024] ;
  for (int i=0; i < 1024; i++) thread_id[i] = -1;
  //#pragma omp target map (tofrom: thread_id)
  #pragma omp target parallel for num_threads(1024)
  for (int i=0; i< 1024; i++) {
    if (i >950)
      printf ("Thread: %d\n", omp_get_thread_num());
    thread_id[i] = omp_get_thread_num();
  }
  // SPMD: if (thread_id[1023] == 1023 ) {
  int maxThrd = -1;
  for (int i=0; i < 1024; i++) {
    if (thread_id[i] > maxThrd)
     maxThrd = thread_id[i];
  }
  printf("Max thread id %d\n", maxThrd);
  if (maxThrd == masterWarpThread) {
    printf("Passed\n");
    return 0;
  } else {
    printf("Failed\n");
    return 1;
  }
}

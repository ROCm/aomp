#include <stdio.h>
#include <omp.h>
#include <string.h>
#include <stdlib.h>

int main()
{
  //Determine which GPU type (NVIDIA or AMD)
  char* nvidia= "sm";
  char* aomp_gpu= getenv("AOMP_GPU");
  int isAMDGPU = 1;
  int masterWarpThread = -1;

  if(aomp_gpu && strstr(aomp_gpu, nvidia) != NULL)
    isAMDGPU = 0;

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

  //Determine execution Mode
  if (maxThrd == 1023)
    printf("Running in SPMD Mode\n");
  else
    printf("Running in Generic Mode\n");

  //Verify Results
  int passed = 0;

  //Check SPMD results
  if (maxThrd == 1023)
    passed = 1;
  else{
    //Check generic results
    if(isAMDGPU)
      maxThrd += 64;
    else
      maxThrd += 32;
    if (maxThrd == 1023)
      passed = 1;
  }
  if (passed){
    printf("Passed!\n");
    return 0;
  } else{
    printf("Failed\n");
    return 1;
  }
}

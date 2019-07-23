#include <stdio.h>
#include <omp.h>
#include <string.h>
#include <stdlib.h>
/*Starting with clang version 9 this test is considered to be SPMD mode. Previously, clang 8 considered this to be generic,
so now this test determines the clang version to verify the results.*/

int main()
{
  FILE * file;
  char ln[256];

  //Default Clang version set to 8
  int clangVersion = 8;

  //Determine which Clang version
  system("$AOMP/bin/clang --version > clang.txt");
  file = fopen("clang.txt", "r");
  if (!file)
    printf("Clang version file not found!\n");
  else{
    char *found = NULL;
    while((fgets(ln, 256, file) != NULL)){
       found = strstr(ln, "9.");
       if(found){
         clangVersion = 9;
         printf("Clang version: %s\n", found);
         break;
       }
    }
  }
  fclose(file);
  system("rm clang.txt");
  //Determine which GPU type (NVIDIA or AMD)
  char* nvidia= "sm";
  char* aomp_gpu= getenv("AOMP_GPU");
  int isAMDGPU = 1;
  int masterWarpThread = -1;

  if(aomp_gpu && strstr(aomp_gpu, nvidia) != NULL)
    isAMDGPU = 0;

  if(isAMDGPU && clangVersion < 9)
    masterWarpThread = 959;

  if(!isAMDGPU)
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
  if (maxThrd == masterWarpThread && clangVersion < 9) {
    printf("Passed\n");
    return 0;
  } else if (maxThrd == 1023 && clangVersion > 8){
    printf("Passed\n");
    return 0;
  } else{
    printf("Failed\n");
    return 1;

  }
}

#include <stdlib.h>
#include <stdio.h>

int main(){
  int verbose = 0;
  if(getenv("VERBOSE_PRINT"))
    verbose = 1;

  if(!verbose){
    int errors = system("diff expected.txt run.log");
    //system("diffrun.log");
    if (errors){
      printf("\nFail! Run.log does not match expected output.\n");
      return 1;
    }
  }
  if(!verbose)
    printf("Passed!\n");
  else
    printf("\nVERBOSE_PRINT does not verify output. Return 0 by default.\n");
  return 0;
}

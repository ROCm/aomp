#include <stdlib.h>
#include <stdio.h>

int main(){
  // Grep returns 0 when string is found and a nonzero otherwise.
  int errors = system("cat run.log | grep 'NUMA domain does not exist in topology'");
  int warnings = system("cat run.log | grep -i warning");
  if (!errors || !warnings){
    printf("\nFail! Run.log does not match expected output.\n");
    return 1;
  }
  printf("Passed!\n");
  return 0;
}

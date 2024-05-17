#include <stdlib.h>
#include <stdio.h>

int main(){
  system("echo Sorted Log:");
  system("sort -n run.log -o sorted.log; cat sorted.log");
  int errors = system("diff expected.txt sorted.log");
  if (errors){
    printf("\nFail! Run.log does not match expected output.\n");
    return 1;
  }
  printf("Passed!\n");
  return 0;
}

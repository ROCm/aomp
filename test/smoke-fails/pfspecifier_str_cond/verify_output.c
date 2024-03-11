#include <stdlib.h>
#include <stdio.h>

int main(){
  int errors = system("diff expected.txt run.log");
  if (errors){
    printf("\nFail! Run.log does not match expected output.\n");
    return 1;
  }
  printf("Passed!\n");
  return 0;
}

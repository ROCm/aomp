#include <cstdio>

extern int hipMallocOmpTarget1();
extern int hipMallocOmpTarget2();

int main() {
  int rc = 0;
  rc = hipMallocOmpTarget1();
  if (rc !=0){
    printf("\nTest hipMallocOmpTarget1 failed\n");
    return rc;
  }
  hipMallocOmpTarget2();
  if (rc !=0){
    printf("\nTest hipMallocOmpTarget2 failed\n");
    return rc;
  }
  printf("Passed \n");
  return 0;
}


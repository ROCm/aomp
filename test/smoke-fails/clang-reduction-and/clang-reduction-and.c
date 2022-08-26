#include <stdio.h>

int main() {
  char result = 1;
  char modifier = 0;
  #pragma omp target teams distribute reduction(&&:result) map(tofrom: result)
  for (int x = 0; x < 10; ++x) {
      result = result && modifier;
  }
  if(result == modifier){
    printf("PASS\n");
    return 0;
  }else{
    printf("FAIL\n");
    return 1;
  }
}

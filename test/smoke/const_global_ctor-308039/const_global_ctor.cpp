#include <cmath>
#include <iostream>

#pragma omp declare target
static  const double smallx            = 1.0e-5;
static  const double log_smallx        = log2(smallx);
#pragma omp end declare target

int main(){

  printf("Success\n");
  return 0;
}

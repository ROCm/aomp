#include <cmath>
#include <stdio.h>

int main(int argc, char ** argv) {
  double x0 = 1.0;
  double sumx = 0.0;
  double n = 1000.0;

  #pragma omp target enter data map(to:x0)

  #pragma omp target teams distribute parallel for map(tofrom: sumx) reduction(+:sumx)
  for (std::size_t i = 0; i < n; ++i ) {
    sumx += x0;
  }

  #pragma omp target exit data map(delete: x0)
  if(sumx != n){
    printf("Fail\n");
    return 1;
  }

  else
    printf("Success!\n");
  return 0;
}


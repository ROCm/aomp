#include <cstdio>

#include <cstdlib>


 

#define NUM 1024


 

int main() {

  int *c= new int[NUM];

  for (int i = 0; i < NUM; i++) {

    c[i] = 1;

  }

#pragma omp target teams distribute parallel for map(tofrom: c[0:NUM])

  for (int i = 0; i < NUM; i++) {

    ++c[i];

  }

  int sum = 0;

  for (int i = 0; i < NUM; i++) {

    sum += c[i];

  }

  // CHECK: Sum = 2048

  printf("Sum = %d\n", sum);

  return 0;

}

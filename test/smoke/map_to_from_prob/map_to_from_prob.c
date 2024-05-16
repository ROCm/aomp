#include <stdio.h>
#include <omp.h>

int main(void) {

double vectors[120];
double blocks[120];
double * vector_base = &vectors[0];
double * __restrict__ vector_grid = vector_base;

#pragma omp target map(to:blocks[0:120], vector_base[0:120]) //map(from:vector_grid[50:20])
#pragma omp teams distribute
  for (int i = 0 ; i < 100 ; i++) {
    //vector_grid[i] = vector_base[i] + blocks[i];
      vector_base[i] = blocks[i];
  }
  printf("Succeeded\n");
  return 0;
}


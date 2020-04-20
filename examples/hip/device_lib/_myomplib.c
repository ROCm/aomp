#include <omp.h>
#pragma omp declare target
void inc_omp(int i, int *array) { array[i]++; }
void dec_omp(int i, int *array) { array[i]--; }
#pragma omp end declare target

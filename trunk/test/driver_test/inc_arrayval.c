#pragma omp declare target
void inc_arrayval(int i, int *array) { array[i]++; }
#pragma omp end declare target

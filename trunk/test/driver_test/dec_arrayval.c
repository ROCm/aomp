#pragma omp declare target
void dec_arrayval(int i, int *array) { array[i]--; }
#pragma omp end declare target

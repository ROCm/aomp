#include <stdio.h>
#include <omp.h>

int main()
{
#pragma omp target parallel num_threads(1)
    printf ("Thread: %d\n", omp_get_thread_num());

  return 0;
}


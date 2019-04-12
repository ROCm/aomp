#include <stdio.h>
#include <omp.h>

int main()
{
int thread_id = -1;
//#pragma omp target map (tofrom: thread_id)
#pragma omp target parallel num_threads(1) map(tofrom: thread_id)
{
    printf ("Thread: %d\n", omp_get_thread_num());
    thread_id = omp_get_thread_num();
}
  return (thread_id == 0)? 0:1;
}


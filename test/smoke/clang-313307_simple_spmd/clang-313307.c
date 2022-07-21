#include <stdio.h>
#include <omp.h>

int main(){
    int nthreads_a;
    int nteams_a;
#pragma omp target parallel map(nteams_a, nthreads_a)
{
    nteams_a = omp_get_num_teams();
    nthreads_a = omp_get_num_threads();
}

    printf("hello %d %d\n", nteams_a, nthreads_a);

    if (nteams_a != 1 || nthreads_a != 256) return 1; 
    return 0;
}


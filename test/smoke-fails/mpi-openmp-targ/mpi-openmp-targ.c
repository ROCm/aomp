#include <mpi.h>
#include "omp.h"
#include <stdio.h>
int main(int argc, char *argv[]) {
    int numranks, rank;
    MPI_Init(&argc,&argv);
    MPI_Comm_size(MPI_COMM_WORLD,&numranks);
    MPI_Comm_rank(MPI_COMM_WORLD,&rank);
    printf ("Number of Ranks= %d My rank= %d\n", numranks,rank);
#pragma omp target 
    {
       printf("hello from rank %d gpu %d\n", rank, omp_get_device_num());
    }
    MPI_Finalize();

    return 0;
}

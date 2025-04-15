#include "omp.h"
#include <mpi.h>
#include <EmissaryMPI.h>
#include <stdio.h>

#define _K 10

int main(int argc, char *argv[])
{
  int rank, numranks;
  MPI_Init(&argc, &argv);
  MPI_Comm_size(MPI_COMM_WORLD, &numranks);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm _mpi_comm = MPI_COMM_WORLD;
  MPI_Datatype _mpi_int = MPI_INT;
  MPI_Op _mpi_sum = MPI_SUM;
  int local_val = _K;
  int final_val = 0;

#pragma omp target map(to: local_val) map(from: final_val)
  {
    MPI_Reduce(&local_val, &final_val, 1, _mpi_int, _mpi_sum, 0, _mpi_comm);
  }

  MPI_Finalize();
  if (rank == 0)
  {
    printf("reduced value for rank %d: %d\n", rank, final_val);
    int expected_val;
    expected_val = numranks * _K;
    if (final_val == expected_val)
      return 0;
    else
      return 1;
  }

  return 0;
}

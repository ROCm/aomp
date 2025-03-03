#include "omp.h"
#include <EmissaryMPI.h>
#include <mpi.h>
#include <stdio.h>

#define _SECRET -123

int main(int argc, char *argv[]) {
  int numranks, rank;
  MPI_Init(&argc, &argv);
  MPI_Comm_size(MPI_COMM_WORLD, &numranks);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm _mpi_comm = MPI_COMM_WORLD;
  MPI_Datatype _mpi_int = MPI_INT;
  int send_recv_buffer[2];
  printf("Number of Ranks= %d My rank= %d buff_addr:%p\n", numranks, rank,
         send_recv_buffer);

#pragma omp target
  {
    if (rank == 0) {
      send_recv_buffer[0] = _SECRET;
      MPI_Send(&send_recv_buffer[0], 1, _mpi_int, 1, 0, _mpi_comm);
    }
    if (rank == 1) {
      MPI_Recv(&send_recv_buffer[0], 1, _mpi_int, 0, 0, _mpi_comm,
               MPI_STATUS_IGNORE);
    }
  }

  MPI_Finalize();
  if (rank == 0) {
    printf("rank 0 sent %d\n", send_recv_buffer[0]);
    return 0;
  }
  if (rank == 1) {
    printf("rank 1 received %d\n", send_recv_buffer[0]);
    if (send_recv_buffer[0] == _SECRET)
      return 0;
    else
      return 1;
  }

  return 0;
}

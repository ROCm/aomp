/* simple OMP offload kernel from Jeff Sandoval 9/22/2020 */
/* Received from John LeRoy Vogt at HPE 2022-01-25 */

#include <omp.h>
#include <iostream>
#include <cmath>
#include <cstdlib>

#ifdef USE_MPI
#include "mpi.h"
#endif
#include <chrono>
#include <thread>

int main(int argc, char* argv[]) {

    int delay = 20;
    int iters = 1000;

    if (argc > 1) {
	delay = atoi(argv[1]);
    }
    if (argc > 2) {
	iters = atoi(argv[2]);
    }
#ifdef USE_MPI
    int ierr = MPI_Init(&argc, &argv);
    if (ierr != MPI_SUCCESS) {
        fprintf(stderr,"Error: MPI_init() failed: %d\n", ierr);
    }
    int rank;
    ierr = MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    if (ierr != MPI_SUCCESS) {
        fprintf(stderr, "Error: MPI_init() failed: %d\n", ierr);
    }
    int num_ranks;
    ierr = MPI_Comm_size(MPI_COMM_WORLD, &num_ranks);
    if (ierr != MPI_SUCCESS) {
        fprintf(stderr, "Error: MPI_init() failed: %d\n", ierr);
    }
#endif
    fprintf(stderr, "max number of threads %d\n", omp_get_max_threads());
    if (delay > 0) {
        fprintf(stderr, "sleeping for %d seconds\n", delay);
        std::this_thread::sleep_for(std::chrono::seconds(delay));
        fprintf(stderr, "continuing\n");
    }

  int runningOnGPU = 0;
  int checkVal=-1;
  const int nCells=1000000;
  double* m_gate = (double*)calloc(nCells,sizeof(double));
  double* Vm = (double*)calloc(nCells,sizeof(double));

  const double Mhu_a[] = { 9.9632117206253790e-01,  4.0825738726469545e-02,  6.3401613233199589e-04,  4.4158436861700431e-06,  1.1622058324043520e-08,  1.0000000000000000e+00,  4.0568375699663400e-02,  6.4216825832642788e-04,  4.2661664422410096e-06,  1.3559930396321903e-08, -1.3573468728873069e-11, -4.2594802366702580e-13,  7.6779952208246166e-15,  1.4260675804433780e-16, -2.6656212072499249e-18};

  const double Tau_a[] = {1.7765862602413648e+01*0.02,  5.0010202770602419e-02*0.02, -7.8002064070783474e-04*0.02, -6.9399661775931530e-05*0.02,  1.6936588308244311e-06*0.02,  5.4629017090963798e-07*0.02, -1.3805420990037933e-08*0.02, -8.0678945216155694e-10*0.02,  1.6209833004622630e-11*0.02,  6.5130101230170358e-13*0.02, -6.9931705949674988e-15*0.02, -3.1161210504114690e-16*0.02,  5.0166191902609083e-19*0.02,  7.8608831661430381e-20*0.02,  4.3936315597226053e-22*0.02, -7.0535966258003289e-24*0.02, -9.0473475495087118e-26*0.02, -2.9878427692323621e-28*0.02,  1.0000000000000000e+00};
      
#pragma omp target enter data map(to: m_gate[:nCells])
#pragma omp target enter data map(to: Vm[:nCells])

  for (int itime=0; itime<iters; itime++) {
    #pragma omp target teams distribute parallel for thread_limit(128) map(from:checkVal)
    for (int ii=0; ii<nCells; ii++) {
      double sum1,sum2;
      const double x = Vm[ii];
      const int Mhu_l = 10;
      const int Mhu_m = 5;

      sum1 = 0;
      for (int j = Mhu_m-1; j >= 0; j--)
         sum1 = Mhu_a[j] + x*sum1;

      sum2 = 0;
      int k = Mhu_m + Mhu_l - 1;
      for (int j = k; j >= Mhu_m; j--)
         sum2 = Mhu_a[j] + x * sum2;
      double mhu = sum1/sum2;

      const int Tau_m = 18;
     sum1 = 0;
      for (int j = Tau_m-1; j >= 0; j--)
         sum1 = Tau_a[j] + x*sum1;

      double tauR = sum1;
      m_gate[ii] += (mhu - m_gate[ii])*(1-exp(-tauR));
      if (ii == 0)
         checkVal=(int) (1000000.0 * m_gate[ii]);
    }
  }

  /* Test if GPU is available using OpenMP4.5 */
#pragma omp target map(from:runningOnGPU)
  {
    if (omp_is_initial_device() == 0)
      runningOnGPU = 1;
  }
  /* If still running on CPU, GPU must not be available */
  if (runningOnGPU && (iters != 1000 || checkVal == 996321) ) {
    printf("PASS\n");
  } else {
    printf("FAIL\n");
  }

#ifdef USE_MPI
    ierr = MPI_Finalize();
    if (ierr != MPI_SUCCESS) {
        fprintf(stderr, "Error: MPI_Finalize() failed: %d\n", ierr);
    }
#endif

  return 0;
}



#include <cstdlib>
#include <cmath>
#include <stdio.h>
#include <sys/time.h>
#include <unistd.h>
#include <stdlib.h>
#include <stdarg.h>

/* Returns time in seconds (double) since the first call to secs_elapsed
 * (i.e., the first call returns 0.0).
 */
double secs_elapsed( void )
{
        static double base_time = -1;
        struct timeval ts;
        int status;
        double new_time;

        /* Get wall-clock time */
        /* status = getclock( CLOCK_REALTIME, &ts ); */
        status = gettimeofday( &ts, NULL );

        /* Return 0.0 on error */
        if( status != 0 ) return 0.0;

        /* Converst structure to double (in seconds ) (a large number) */
        new_time = (double)ts.tv_sec + (double)ts.tv_usec * 1e-6;

        /* If first time called, set base_time
         * Note: Lock shouldn't be needed, since even if multiple
         *       threads initialize this, it will be to basically
         *       the same value.
         */
        if (base_time < 0)
            base_time = new_time;

        /* Returned offset from first time called */
        return (new_time - base_time);
}

/* Works like printf, except prefixes wall-clock time (using secs_elapsed)
 * and writes to stderr.  Also flushes stdout, so messages stay
 * in reasonable order.
 */
void printf_timestamp (const char * fmt, ...)
{
    va_list args;
    char buf[4096];

    /* Get wall-clock time since first call to secs_elapsed */
    double sec = secs_elapsed();

    /* Flush stdout, so message appear in reasonable order */
    fflush (stdout);

    /* Print out passed message to big buffer*/
    va_start (args, fmt);
    vsnprintf (buf, sizeof(buf), fmt, args);

    /* Print out timestamp with buffer*/
    fprintf (stdout, "%7.3f: %s", sec, buf);

    va_end (args);
}


__global__ void doRushLarsen(double* m_gate, const int nCells, const double* Vm) {
   int ii = blockIdx.x*blockDim.x + threadIdx.x;
   if (ii > nCells) { return; }

   double sum1,sum2;
   const double x = Vm[ii];
   const int Mhu_l = 10;
   const int Mhu_m = 5;
   const double Mhu_a[] = { 9.9632117206253790e-01,  4.0825738726469545e-02,  6.3401613233199589e-04,  4.4158436861700431e-06,  1.1622058324043520e-08,  1.0000000000000000e+00,  4.0568375699663400e-02,  6.4216825832642788e-04,  4.2661664422410096e-06,  1.3559930396321903e-08, -1.3573468728873069e-11, -4.2594802366702580e-13,  7.6779952208246166e-15,  1.4260675804433780e-16, -2.6656212072499249e-18};
      
   sum1 = 0;
   for (int j = Mhu_m-1; j >= 0; j--)
     sum1 = Mhu_a[j] + x*sum1;
      
   sum2 = 0;
   int k = Mhu_m + Mhu_l - 1;
   for (int j = k; j >= Mhu_m; j--)
     sum2 = Mhu_a[j] + x * sum2;
   double mhu = sum1/sum2;

   const int Tau_m = 18;
   const double Tau_a[] = {1.7765862602413648e+01*0.02,  5.0010202770602419e-02*0.02, -7.8002064070783474e-04*0.02, -6.9399661775931530e-05*0.02,  1.6936588308244311e-06*0.02,  5.4629017090963798e-07*0.02, -1.3805420990037933e-08*0.02, -8.0678945216155694e-10*0.02,  1.6209833004622630e-11*0.02,  6.5130101230170358e-13*0.02, -6.9931705949674988e-15*0.02, -3.1161210504114690e-16*0.02,  5.0166191902609083e-19*0.02,  7.8608831661430381e-20*0.02,  4.3936315597226053e-22*0.02, -7.0535966258003289e-24*0.02, -9.0473475495087118e-26*0.02, -2.9878427692323621e-28*0.02,  1.0000000000000000e+00};
      
   sum1 = 0;
   for (int j = Tau_m-1; j >= 0; j--)
     sum1 = Tau_a[j] + x*sum1;

   double tauR = sum1;
   m_gate[ii] += (mhu - m_gate[ii])*(1-exp(-tauR));
}


int main(int argc, char* argv[]) {
  double after1;
  printf_timestamp("Starting 10000 CUDA GPU loops doing Rush Larsen\n"); 
  const int nCells=1000000;
  double* m_gate = (double*)calloc(nCells,sizeof(double));
  double* Vm = (double*)calloc(nCells,sizeof(double));

  double* c_m_gate;
  cudaMalloc(&c_m_gate, sizeof(double)*nCells);
  cudaMemcpy(c_m_gate, m_gate, sizeof(double)*nCells, cudaMemcpyHostToDevice);
  double* c_Vm;
  cudaMalloc(&c_Vm, sizeof(double)*nCells);
  cudaMemcpy(c_Vm, Vm, sizeof(double)*nCells, cudaMemcpyHostToDevice);

  dim3 gridSize, blockSize;
  blockSize.x=512; blockSize.y=1; blockSize.z=1;
  gridSize.x = (nCells + blockSize.x-1) / blockSize.x; gridSize.y=1; gridSize.z=1;
  for (int itime=0; itime<10000; itime++) {
    if ((itime==1) || ((itime % 1000) == 0))
       printf_timestamp("Starting iteration %i\n", itime);
    if (itime==1)
       after1=secs_elapsed();
    doRushLarsen<<<gridSize, blockSize>>>(c_m_gate, nCells, c_Vm);
    cudaDeviceSynchronize();
  }
  printf_timestamp("Done, took %.2lf seconds after 1st iter\n", secs_elapsed()-after1);

  return 0;
}


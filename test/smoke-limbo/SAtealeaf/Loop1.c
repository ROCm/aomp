// Standalone program, highest time-burning loop in 518.tealeaf.
// 518.tealeaf_t/build/build_base_clang_omp_target_gfx90a.0000/2d/c_kernels/cg.c
// cg_calc_w line 184

#include <stdio.h>
#include <stdarg.h>
#include <stdlib.h>
#include <omp.h>


int main(int argc, char** argv)
{
    double * w;
    double * p;
    double * kx;
    double * ky;
    double pw_temp;
    int x, y, halo_depth;

    x = 8192;
    y = 8192;

    w  = (double*)malloc(sizeof(double)*x*y);
    p  = (double*)malloc(sizeof(double)*x*y);
    kx = (double*)malloc(sizeof(double)*x*y);
    ky = (double*)malloc(sizeof(double)*x*y);

    srand(123456789);
    for(int jj = 0; jj < y; ++jj) {
        for(int kk = 0; kk < x; ++kk) {
            const int index = kk + jj*x;
            w [index] = (double)rand()/RAND_MAX;
            p [index] = (double)rand()/RAND_MAX;
            kx[index] = (double)rand()/RAND_MAX;
            ky[index] = (double)rand()/RAND_MAX;
        }
    }

    halo_depth = 2;
    int xy = x*y;

    for (int i=0; i<5; i++) {
       double t0 = omp_get_wtime();
       #pragma omp target teams distribute parallel for simd map(tofrom:pw_temp, w[:xy], p[:xy], kx[:xy], ky[:xy]) reduction(+:pw_temp) collapse(2)
       for(int jj = halo_depth; jj < y-halo_depth; ++jj) {
          for(int kk = halo_depth; kk < x-halo_depth; ++kk) {
             const int index = kk + jj*x;
             const double smvp = (1.0 + (kx[index+1]+kx[index])\
                           + (ky[index+x]+ky[index]))*p[index]\
                           - (kx[index+1]*p[index+1]+kx[index]*p[index-1])\
                           - (ky[index+x]*p[index+x]+ky[index]*p[index-x]);

             w[index] = smvp;
             pw_temp += w[index]*p[index];
          }
       }

       double elapsed = omp_get_wtime() - t0;
       printf ("elapsed = %f\n", elapsed);
       printf ("pw_temp= %f\n",pw_temp);
       printf ("w,p=%f, %f\n\n",w[0],p[0]);
    }
    printf ("Success.\n");
}


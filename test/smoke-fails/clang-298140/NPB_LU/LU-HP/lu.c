//-------------------------------------------------------------------------//
//                                                                         //
//  This benchmark is a serial C version of the NPB LU code. This C        //
//  version is developed by the Center for Manycore Programming at Seoul   //
//  National University and derived from the serial Fortran versions in    //
//  "NPB3.3-SER" developed by NAS.                                         //
//                                                                         //
//  Permission to use, copy, distribute and modify this software for any   //
//  purpose with or without fee is hereby granted. This software is        //
//  provided "as is" without express or implied warranty.                  //
//                                                                         //
//  Information on NPB 3.3, including the technical report, the original   //
//  specifications, source code, results and information on how to submit  //
//  new results, is available at:                                          //
//                                                                         //
//           http://www.nas.nasa.gov/Software/NPB/                         //
//                                                                         //
//  Send comments or suggestions for this C version to cmp@aces.snu.ac.kr  //
//                                                                         //
//          Center for Manycore Programming                                //
//          School of Computer Science and Engineering                     //
//          Seoul National University                                      //
//          Seoul 151-744, Korea                                           //
//                                                                         //
//          E-mail:  cmp@aces.snu.ac.kr                                    //
//                                                                         //
//-------------------------------------------------------------------------//

//-------------------------------------------------------------------------//
// Authors: Sangmin Seo, Jungwon Kim, Jun Lee, Jeongho Nah, Gangwon Jo,    //
//          and Jaejin Lee                                                 //
//-------------------------------------------------------------------------//

//-------------------------------------------------------------------------//
////                                                                         //
////  The OpenACC C version of the NAS LU code is developed by the           //
////  HPCTools Group of University of Houston and derived from the serial    //
////  C version developed by SNU and Fortran versions in "NPB3.3-SER"        //
////  developed by NAS.                                                      //
////                                                                         //
////  Permission to use, copy, distribute and modify this software for any   //
////  purpose with or without fee is hereby granted. This software is        //
////  provided "as is" without express or implied warranty.                  //
////                                                                         //
////  Send comments or suggestions for this OpenACC version to               //
////                      hpctools@cs.uh.edu                                 //
////
////  Information on NPB 3.3, including the technical report, the original   //
////  specifications, source code, results and information on how to submit  //
////  new results, is available at:                                          //
////                                                                         //
////           http://www.nas.nasa.gov/Software/NPB/                         //
////                                                                         //
////-------------------------------------------------------------------------//
//
////-------------------------------------------------------------------------//
//// Authors: Rengan Xu, Sunita Chandrasekaran, Barbara Chapman              //
////-------------------------------------------------------------------------//

//---------------------------------------------------------------------
//   program applu
//---------------------------------------------------------------------

//---------------------------------------------------------------------
//
//   driver for the performance evaluation of the solver for
//   five coupled parabolic/elliptic partial differential equations.
//
//---------------------------------------------------------------------

#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include "applu.incl"
#include "timers.h"
#include "print_results.h"
#include <omp.h>

//---------------------------------------------------------------------
// grid
//---------------------------------------------------------------------
/* common/cgcon/ */
double dxi, deta, dzeta;
double tx1, tx2, tx3;
double ty1, ty2, ty3;
double tz1, tz2, tz3;
int nx, ny, nz;
int nx0, ny0, nz0;
int ist, iend;
int jst, jend;
int ii1, ii2;
int ji1, ji2;
int ki1, ki2;

//---------------------------------------------------------------------
// dissipation
//---------------------------------------------------------------------
/* common/disp/ */
double dx1, dx2, dx3, dx4, dx5;
double dy1, dy2, dy3, dy4, dy5;
double dz1, dz2, dz3, dz4, dz5;
double dssp;

//---------------------------------------------------------------------
// field variables and residuals
// to improve cache performance, second two dimensions padded by 1 
// for even number sizes only.
// Note: corresponding array (called "v") in routines blts, buts, 
// and l2norm are similarly padded
//---------------------------------------------------------------------
/* common/cvar/ */
double u[5][ISIZ3][ISIZ2/2*2+1][ISIZ1/2*2+1];
double rsd[5][ISIZ3][ISIZ2/2*2+1][ISIZ1/2*2+1];
double frct[5][ISIZ3][ISIZ2/2*2+1][ISIZ1/2*2+1];
double flux_G[5][ISIZ3][ISIZ2][ISIZ1];
double qs   [ISIZ3][ISIZ2/2*2+1][ISIZ1/2*2+1];
double rho_i[ISIZ3][ISIZ2/2*2+1][ISIZ1/2*2+1];

//---------------------------------------------------------------------
// output control parameters
//---------------------------------------------------------------------
/* common/cprcon/ */
int ipr, inorm;

//---------------------------------------------------------------------
// newton-raphson iteration control parameters
//---------------------------------------------------------------------
/* common/ctscon/ */
double dt, omega, tolrsd[5], rsdnm[5], errnm[5], frc, ttotal;
int itmax, invert;

/* common/cjac/ */
double a[5][5][ISIZ1*ISIZ2];
double b[5][5][ISIZ1*ISIZ2];
double c[5][5][ISIZ1*ISIZ2];
double d[5][5][ISIZ1*ISIZ2];

int np[ISIZ1+ISIZ2+ISIZ3-5];
int indxp[ISIZ1+ISIZ2+ISIZ3-5][ISIZ1*ISIZ2*3/4];
int jndxp[ISIZ1+ISIZ2+ISIZ3-5][ISIZ1*ISIZ2*3/4];
double tmat[5][5][ISIZ1*ISIZ2*3/4];
double tv[5][ISIZ1*ISIZ2*3/4];
double utmp_G[6][ISIZ2][ISIZ1][ISIZ3];
double rtmp_G[5][ISIZ2][ISIZ1][ISIZ3];

//---------------------------------------------------------------------
// coefficients of the exact solution
//---------------------------------------------------------------------
/* common/cexact/ */
double ce[5][13];


//---------------------------------------------------------------------
// timers
//---------------------------------------------------------------------
/* common/timer/ */
double maxtime;
logical timeron;

//---------------------------------------------------------------------
// openMP 4.5
//---------------------------------------------------------------------

void omp_device_mem_init() {
  #pragma omp target enter data \
          map (alloc: u) \
          map (alloc: a) \
          map (alloc: b) \
          map (alloc: c) \
          map (alloc: d) \
          map (alloc: flux_G) \
          map (alloc: indxp) \
          map (alloc: jndxp) \
          map (alloc: rho_i) \
          map (alloc: frct) \
          map (alloc: qs) \
          map (alloc: rsd) \
          map (alloc: tmat) \
          map (alloc: tv) \
          map (alloc: utmp_G) \
          map (alloc: rtmp_G) \
          map (alloc: rsdnm) \
          map (to: ce)
}

void omp_device_mem_delete() {
  #pragma omp target exit data \
          map (delete: u) \
          map (delete: a) \
          map (delete: b) \
          map (delete: c) \
          map (delete: d) \
          map (delete: flux_G) \
          map (delete: indxp) \
          map (delete: jndxp) \
          map (delete: rho_i) \
          map (delete: frct) \
          map (delete: qs) \
          map (delete: rsd) \
          map (delete: tmat) \
          map (delete: tv) \
          map (delete: utmp_G) \
          map (delete: rtmp_G) \
          map (delete: ce) \
          map (from: rsdnm) 
} // TODO: Change rsdnm to from again

int main(int argc, char *argv[])
{
  char Class;
  logical verified;
  double mflops;

  double t, tmax, trecs[t_last+1];
  int i;
  char *t_names[t_last+1];

  //---------------------------------------------------------------------
  // Setup info for timers
  //---------------------------------------------------------------------
  FILE *fp;
  if ((fp = fopen("timer.flag", "r")) != NULL) {
    timeron = true;
    t_names[t_total] = "total";
    t_names[t_rhsx] = "rhsx";
    t_names[t_rhsy] = "rhsy";
    t_names[t_rhsz] = "rhsz";
    t_names[t_rhs] = "rhs";
    t_names[t_jacld] = "jacld";
    t_names[t_blts] = "blts";
    t_names[t_jacu] = "jacu";
    t_names[t_buts] = "buts";
    t_names[t_add] = "add";
    t_names[t_l2norm] = "l2norm";
    fclose(fp);
  } else {
    timeron = false;
  }

  //---------------------------------------------------------------------
  // read input data
  //---------------------------------------------------------------------
  read_input();
  //---------------------------------------------------------------------
  // set up domain sizes
  //---------------------------------------------------------------------
  domain();

  //---------------------------------------------------------------------
  // set up coefficients
  //---------------------------------------------------------------------
  setcoeff();

  omp_device_mem_init();

  //---------------------------------------------------------------------
  // set the boundary values for dependent variables
  //---------------------------------------------------------------------
  setbv();

  //---------------------------------------------------------------------
  // set the initial values for dependent variables
  //---------------------------------------------------------------------
  setiv();
#pragma omp target update to(u[0:5][0:ISIZ3][0:(ISIZ2/2*2+1)][0:(ISIZ1/2*2+1)])
  //---------------------------------------------------------------------
  // compute the forcing term based on prescribed exact solution
  //---------------------------------------------------------------------
  erhs();

  //---------------------------------------------------------------------
  // perform one SSOR iteration to touch all pages
  //---------------------------------------------------------------------
  ssor(1);

  //---------------------------------------------------------------------
  // reset the boundary and initial values
  //---------------------------------------------------------------------
  setbv();
  setiv();
 #pragma omp target update to(u[0:5][0:ISIZ3][0:(ISIZ2/2*2+1)][0:(ISIZ1/2*2+1)])

  //---------------------------------------------------------------------
  // perform the SSOR iterations
  //---------------------------------------------------------------------
  ssor(itmax);


  //---------------------------------------------------------------------
  // compute the solution error
  //---------------------------------------------------------------------
#pragma omp target update from(u[0:5][0:ISIZ3][0:(ISIZ2/2*2+1)][0:(ISIZ1/2*2+1)])
  error();

  //---------------------------------------------------------------------
  // compute the surface integral
  //---------------------------------------------------------------------
  pintgr();

  omp_device_mem_delete();

  //---------------------------------------------------------------------
  // verification test
  //---------------------------------------------------------------------
  verify ( rsdnm, errnm, frc, &Class, &verified );
  mflops = (double)itmax * (1984.77 * (double)nx0
      * (double)ny0
      * (double)nz0
      - 10923.3 * pow(((double)(nx0+ny0+nz0)/3.0), 2.0) 
      + 27770.9 * (double)(nx0+ny0+nz0)/3.0
      - 144010.0)
    / (maxtime*1000000.0);

  print_results("LU", Class, nx0,
                ny0, nz0, itmax,
                maxtime, mflops, "          floating point", verified, 
                NPBVERSION, COMPILETIME, CS1, CS2, CS3, CS4, CS5, CS6, 
                "(none)");

  //---------------------------------------------------------------------
  // More timers
  //---------------------------------------------------------------------
  if (timeron) {
    for (i = 1; i <= t_last; i++) {
      trecs[i] = timer_read(i);
    }
    tmax = maxtime;
    if (tmax == 0.0) tmax = 1.0;

    printf("  SECTION     Time (secs)\n");
    for (i = 1; i <= t_last; i++) {
      printf("  %-8s:%9.3f  (%6.2f%%)\n",
          t_names[i], trecs[i], trecs[i]*100./tmax);
      if (i == t_rhs) {
        t = trecs[t_rhsx] + trecs[t_rhsy] + trecs[t_rhsz];
        printf("     --> %8s:%9.3f  (%6.2f%%)\n", "sub-rhs", t, t*100./tmax);
        t = trecs[i] - t;
        printf("     --> %8s:%9.3f  (%6.2f%%)\n", "rest-rhs", t, t*100./tmax);
      }
    }
  }

  return 0;
}


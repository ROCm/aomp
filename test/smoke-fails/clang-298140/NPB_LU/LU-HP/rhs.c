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

#include "applu.incl"
#include "timers.h"
#include <stdio.h>
#include <omp.h>

//---------------------------------------------------------------------
// compute the right hand sides
//---------------------------------------------------------------------
void rhs()
{
  //---------------------------------------------------------------------
  // local variables
  //---------------------------------------------------------------------
  int i, j, k, m;
  double q;
  double tmp;
  double u21, u31, u41;
  double u21i, u31i, u41i, u51i;
  double u21j, u31j, u41j, u51j;
  double u21k, u31k, u41k, u51k;
  double u21im1, u31im1, u41im1, u51im1;
  double u21jm1, u31jm1, u41jm1, u51jm1;
  double u21km1, u31km1, u41km1, u51km1;
  unsigned num_workers3 = 0;
  unsigned num_workers2 = 0;

    if (timeron) timer_start(t_rhs);
#pragma omp target data \
 map (alloc: frct, rsd, rtmp_G, utmp_G, rho_i, flux_G, qs, u)
{
#ifndef CRPL_COMP
#elif CRPL_COMP == 0
    #pragma omp target teams map (alloc: frct, u, rho_i, qs) \
    num_teams(nz) //thread_limit(128)
    {
    #pragma omp distribute parallel for collapse (3) private(tmp)
#endif
    for (k = 0; k < nz; k++) {
      for (j = 0; j < ny; j++) {
        //#pragma simd private (tmp) simdlen(32)
        for (i = 0; i < nx; i++) {
          // for (m = 0; m < 5; m++) { LOOP UNROLLING
            rsd[0][k][j][i] = - frct[0][k][j][i];
            rsd[1][k][j][i] = - frct[1][k][j][i];
            rsd[2][k][j][i] = - frct[2][k][j][i];
            rsd[3][k][j][i] = - frct[3][k][j][i];
            rsd[4][k][j][i] = - frct[4][k][j][i];
          //}
          tmp = 1.0 / u[0][k][j][i];
          rho_i[k][j][i] = tmp;
          qs[k][j][i] = 0.50 * (  u[1][k][j][i] * u[1][k][j][i]
                                                             + u[2][k][j][i] * u[2][k][j][i]
                                                                                          + u[3][k][j][i] * u[3][k][j][i] )
                                                                                          * tmp;
        }
      }
    }
  }


    if (timeron) timer_start(t_rhsx);
    if(((jend-jst+1))<32)
      num_workers3 = (jend-jst+1);
    else
      num_workers3 = 32;
    //---------------------------------------------------------------------
    // xi-direction flux differences
    //---------------------------------------------------------------------
#ifndef CRPL_COMP
#elif CRPL_COMP == 0
    #pragma omp target teams map (alloc: flux_G, u, qs) \
    num_teams(nz-2)
    #pragma omp distribute parallel for collapse (2) private (u21, q, k, j, i)
#endif
    for (k = 1; k < nz - 1; k++) {
      for (j = jst; j <= jend; j++) {
        for (i = 0; i < nx; i++) {
          flux_G[0][k][j][i] = u[1][k][j][i];
          u21 = u[1][k][j][i] * rho_i[k][j][i];

          q = qs[k][j][i];

          flux_G[1][k][j][i] = u[1][k][j][i] * u21 + C2 * ( u[4][k][j][i] - q );
          flux_G[2][k][j][i] = u[2][k][j][i] * u21;
          flux_G[3][k][j][i] = u[3][k][j][i] * u21;
          flux_G[4][k][j][i] = ( C1 * u[4][k][j][i] - C2 * q ) * u21;
        }
      }
    }

#ifndef CRPL_COMP
#elif CRPL_COMP == 0
    #pragma omp target teams map (alloc: rsd, flux_G) \
    num_teams(nz-2)
    #pragma omp distribute parallel for collapse(3) 
#endif
    for (k = 1; k < nz - 1; k++) {
      for (j = jst; j <= jend; j++) {
        for (i = ist; i <= iend; i++) {
          for (m = 0; m < 5; m++) {
            rsd[m][k][j][i] =  rsd[m][k][j][i]
                                            - tx2 * ( flux_G[m][k][j][i+1] - flux_G[m][k][j][i-1] );
          }
        }
      }
    }

#ifndef CRPL_COMP 
#elif CRPL_COMP == 0
    #pragma omp target teams map (alloc: u, flux_G, rho_i) \
    num_teams(nz-2) thread_limit(num_workers3)
    #pragma omp distribute parallel for collapse(2) \
            private (k, j, i, tmp, u21i, u31i, u41i, u51i, u21im1, u31im1, u41im1, u51im1)
#endif
    for (k = 1; k < nz - 1; k++) {
      for (j = jst; j <= jend; j++) {
        #pragma omp simd simdlen(32) private (tmp, u21i, u31i, u41i, u51i, u21im1, u31im1, u41im1, u51im1)
        for (i = ist; i < nx; i++) {
          tmp = rho_i[k][j][i];

          u21i = tmp * u[1][k][j][i];
          u31i = tmp * u[2][k][j][i];
          u41i = tmp * u[3][k][j][i];
          u51i = tmp * u[4][k][j][i];

          tmp = rho_i[k][j][i-1];

          u21im1 = tmp * u[1][k][j][i-1];
          u31im1 = tmp * u[2][k][j][i-1];
          u41im1 = tmp * u[3][k][j][i-1];
          u51im1 = tmp * u[4][k][j][i-1];

          flux_G[1][k][j][i] = (4.0/3.0) * tx3 * (u21i-u21im1);
          flux_G[2][k][j][i] = tx3 * ( u31i - u31im1 );
          flux_G[3][k][j][i] = tx3 * ( u41i - u41im1 );
          flux_G[4][k][j][i] = 0.50 * ( 1.0 - C1*C5 )
              * tx3 * ( ( u21i*u21i     + u31i*u31i     + u41i*u41i )
                  - ( u21im1*u21im1 + u31im1*u31im1 + u41im1*u41im1 ) )
                  + (1.0/6.0)
                  * tx3 * ( u21i*u21i - u21im1*u21im1 )
                  + C1 * C5 * tx3 * ( u51i - u51im1 );
        }
      }
    }

#ifndef CRPL_COMP
#elif CRPL_COMP == 0
    #pragma omp target teams map(alloc: rsd, u) \
    num_teams(nz-2)
    #pragma omp distribute parallel for collapse(3)
#endif
    for (k = 1; k < nz - 1; k++) {
      for (j = jst; j <= jend; j++) {
        for (i = ist; i <= iend; i++) {
          rsd[0][k][j][i] = rsd[0][k][j][i]
                                         + dx1 * tx1 * (        u[0][k][j][i-1]
                                                                           - 2.0 * u[0][k][j][i]
                                                                                              +       u[0][k][j][i+1] );
          rsd[1][k][j][i] = rsd[1][k][j][i]
                                         + tx3 * C3 * C4 * ( flux_G[1][k][j][i+1] - flux_G[1][k][j][i] )
                                         + dx2 * tx1 * (        u[1][k][j][i-1]
                                                                           - 2.0 * u[1][k][j][i]
                                                                                              +       u[1][k][j][i+1] );
          rsd[2][k][j][i] = rsd[2][k][j][i]
                                         + tx3 * C3 * C4 * ( flux_G[2][k][j][i+1] - flux_G[2][k][j][i] )
                                         + dx3 * tx1 * (        u[2][k][j][i-1]
                                                                           - 2.0 * u[2][k][j][i]
                                                                                              +       u[2][k][j][i+1] );
          rsd[3][k][j][i] = rsd[3][k][j][i]
                                         + tx3 * C3 * C4 * ( flux_G[3][k][j][i+1] - flux_G[3][k][j][i] )
                                         + dx4 * tx1 * (        u[3][k][j][i-1]
                                                                           - 2.0 * u[3][k][j][i]
                                                                                              +       u[3][k][j][i+1] );
          rsd[4][k][j][i] = rsd[4][k][j][i]
                                         + tx3 * C3 * C4 * ( flux_G[4][k][j][i+1] - flux_G[4][k][j][i] )
                                         + dx5 * tx1 * (        u[4][k][j][i-1]
                                                                           - 2.0 * u[4][k][j][i]
                                                                                              +       u[4][k][j][i+1] );
        }
      }
    }

    //---------------------------------------------------------------------
    // Fourth-order dissipation
    //---------------------------------------------------------------------
    if(((jend-jst+1)/32)<32)
      num_workers2 = (jend-jst+1)/32;
    else
      num_workers2 = 32;

// JOSE TODO how to parallelize this one better?
#ifndef CRPL_COMP
#elif CRPL_COMP == 0
    #pragma omp target teams map (alloc: rsd, u) \
    //num_teams(nz-2) thread_limit(num_workers2)
    #pragma omp distribute
#endif
    for (k = 1; k < nz - 1; k++) {
      //#pragma omp parallel for
      for (j = jst; j <= jend; j++) {
        for (m = 0; m < 5; m++) {
          rsd[m][k][j][1] = rsd[m][k][j][1]
                                         - dssp * ( + 5.0 * u[m][k][j][1]
                                                                       - 4.0 * u[m][k][j][2]
                                                                                          +       u[m][k][j][3] );
          rsd[m][k][j][2] = rsd[m][k][j][2]
                                         - dssp * ( - 4.0 * u[m][k][j][1]
                                                                       + 6.0 * u[m][k][j][2]
                                                                                          - 4.0 * u[m][k][j][3]
                                                                                                             +       u[m][k][j][4] );
        }
      }
    }

// JOSE TODO how to parallelize this one better?
#ifndef CRPL_COMP
#elif CRPL_COMP == 0
    #pragma omp target teams map (alloc: rsd, u) \
    num_teams(nz-2) //thread_limit(num_workers2)
    #pragma omp distribute parallel for private (j, m, i) collapse(3)
#endif
    for (k = 1; k < nz - 1; k++) {
      //#pragma omp parallel for
      for (j = jst; j <= jend; j++) {
        //#pragma omp simd collapse(2)
        for (i = 3; i < nx - 3; i++) {
          for (m = 0; m < 5; m++) {
            rsd[m][k][j][i] = rsd[m][k][j][i]
                                           - dssp * (         u[m][k][j][i-2]
                                                                         - 4.0 * u[m][k][j][i-1]
                                                                                            + 6.0 * u[m][k][j][i]
                                                                                                               - 4.0 * u[m][k][j][i+1]
                                                                                                                                  +       u[m][k][j][i+2] );
          }
        }
      }
    }

#ifndef CRPL_COMP
#elif CRPL_COMP == 0
    #pragma omp target teams map (alloc: rsd, u) \
    num_teams(nz-2) //thread_limit()
    #pragma omp distribute parallel for private(k, j, m) collapse(2)
#endif
    for (k = 1; k < nz - 1; k++) {
      for (j = jst; j <= jend; j++) {
        for (m = 0; m < 5; m++) {
          rsd[m][k][j][nx-3] = rsd[m][k][j][nx-3]
                                            - dssp * (         u[m][k][j][nx-5]
                                                                          - 4.0 * u[m][k][j][nx-4]
                                                                                             + 6.0 * u[m][k][j][nx-3]
                                                                                                                - 4.0 * u[m][k][j][nx-2] );
          rsd[m][k][j][nx-2] = rsd[m][k][j][nx-2]
                                            - dssp * (         u[m][k][j][nx-4]
                                                                          - 4.0 * u[m][k][j][nx-3]
                                                                                             + 5.0 * u[m][k][j][nx-2] );
        }

      }
    }
    
    if (timeron) timer_stop(t_rhsx);

    if (timeron) timer_start(t_rhsy);
    //---------------------------------------------------------------------
    // eta-direction flux differences
    //---------------------------------------------------------------------
    if(((jend-jst+1))<32)
      num_workers3 = (iend-ist+1);
    else
      num_workers3 = 32;

// JOSE TODO how to parallelize this one better?
#ifndef CRPL_COMP
#elif CRPL_COMP == 0
    #pragma omp target teams map (alloc: flux_G, qs, u) \
    num_teams(nz-2) //thread_limit(num_workers2)
    #pragma omp distribute parallel for collapse(3) private(k, i, j, u31, q)
#endif
    for (k = 1; k < nz - 1; k++) {
      for (i = ist; i <= iend; i++) {
        for (j = 0; j < ny; j++) {
          flux_G[0][k][i][j] = u[2][k][j][i];
          u31 = u[2][k][j][i] * rho_i[k][j][i];

          q = qs[k][j][i];

          flux_G[1][k][i][j] = u[1][k][j][i] * u31;
          flux_G[2][k][i][j] = u[2][k][j][i] * u31 + C2 * (u[4][k][j][i]-q);
          flux_G[3][k][i][j] = u[3][k][j][i] * u31;
          flux_G[4][k][i][j] = ( C1 * u[4][k][j][i] - C2 * q ) * u31;
        }
      }
    }
// JOSE TODO how to parallelize this one better
#ifndef CRPL_COMP
#elif CRPL_COMP == 0
    #pragma omp target teams map (alloc: rsd, flux_G) \
    num_teams(nz-2)
    #pragma omp distribute parallel for collapse(3) private(k, i, j)
#endif
    for (k = 1; k < nz - 1; k++) {
      for (i = ist; i <= iend; i++) {
        for (j = jst; j <= jend; j++) {
          for (m = 0; m < 5; m++) {
            rsd[m][k][j][i] =  rsd[m][k][j][i]
                                            - ty2 * ( flux_G[m][k][i][j+1] - flux_G[m][k][i][j-1] );
          }
        }
      }
    }

#ifndef CRPL_COMP
    #pragma omp target teams 
    #pragma omp distribute // \
            private(k, i, j, u21j, u31j, u41j, u51j, tmp, u21jm1, u31jm1, u41jm1, u51jm1 )
#elif CRPL_COMP == 0
#pragma omp target teams map(alloc: u, flux_G, rho_i) \
    num_teams(nz-2)
    #pragma omp distribute parallel for private(k, i, j, u21j, u31j, u41j, u51j, tmp, u21jm1, u31jm1, u41jm1, u51jm1) collapse(3)
#endif
    for (k = 1; k < nz - 1; k++) {
      for (i = ist; i <= iend; i++) {
        for (j = jst; j < ny; j++) { 
          tmp = rho_i[k][j][i];

          u21j = tmp * u[1][k][j][i];
          u31j = tmp * u[2][k][j][i];
          u41j = tmp * u[3][k][j][i];
          u51j = tmp * u[4][k][j][i];

          tmp = rho_i[k][j-1][i];
          u21jm1 = tmp * u[1][k][j-1][i];
          u31jm1 = tmp * u[2][k][j-1][i];
          u41jm1 = tmp * u[3][k][j-1][i];
          u51jm1 = tmp * u[4][k][j-1][i];

          flux_G[1][k][i][j] = ty3 * ( u21j - u21jm1 );
          flux_G[2][k][i][j] = (4.0/3.0) * ty3 * (u31j-u31jm1);
          flux_G[3][k][i][j] = ty3 * ( u41j - u41jm1 );
          flux_G[4][k][i][j] = 0.50 * ( 1.0 - C1*C5 )
              * ty3 * ( ( u21j*u21j     + u31j*u31j     + u41j*u41j )
                  - ( u21jm1*u21jm1 + u31jm1*u31jm1 + u41jm1*u41jm1 ) )
                  + (1.0/6.0)
                  * ty3 * ( u31j*u31j - u31jm1*u31jm1 )
                  + C1 * C5 * ty3 * ( u51j - u51jm1 );
        }
      }
    }

#ifndef CRPL_COMP
#elif CRPL_COMP == 0
    #pragma omp target teams map(alloc: rsd, u, flux_G) \
    num_teams(nz-2)
    #pragma omp distribute parallel for collapse(3)
#endif
    for (k = 1; k < nz - 1; k++) {
      for (i = ist; i <= iend; i++) {
        for (j = jst; j <= jend; j++) {
          rsd[0][k][j][i] = rsd[0][k][j][i]
                                         + dy1 * ty1 * (         u[0][k][j-1][i]
                                                                              - 2.0 * u[0][k][j][i]
                                                                                                 +       u[0][k][j+1][i] );

          rsd[1][k][j][i] = rsd[1][k][j][i]
                                         + ty3 * C3 * C4 * ( flux_G[1][k][i][j+1] - flux_G[1][k][i][j] )
                                         + dy2 * ty1 * (         u[1][k][j-1][i]
                                                                              - 2.0 * u[1][k][j][i]
                                                                                                 +       u[1][k][j+1][i] );

          rsd[2][k][j][i] = rsd[2][k][j][i]
                                         + ty3 * C3 * C4 * ( flux_G[2][k][i][j+1] - flux_G[2][k][i][j] )
                                         + dy3 * ty1 * (         u[2][k][j-1][i]
                                                                              - 2.0 * u[2][k][j][i]
                                                                                                 +       u[2][k][j+1][i] );

          rsd[3][k][j][i] = rsd[3][k][j][i]
                                         + ty3 * C3 * C4 * ( flux_G[3][k][i][j+1] - flux_G[3][k][i][j] )
                                         + dy4 * ty1 * (         u[3][k][j-1][i]
                                                                              - 2.0 * u[3][k][j][i]
                                                                                                 +       u[3][k][j+1][i] );

          rsd[4][k][j][i] = rsd[4][k][j][i]
                                         + ty3 * C3 * C4 * ( flux_G[4][k][i][j+1] - flux_G[4][k][i][j] )
                                         + dy5 * ty1 * (         u[4][k][j-1][i]
                                                                              - 2.0 * u[4][k][j][i]
                                                                                                 +       u[4][k][j+1][i] );
        }
      }
    }

    //---------------------------------------------------------------------
    // fourth-order dissipation
    //---------------------------------------------------------------------
    if(((jend-jst+1)/32)<32)
      num_workers2 = (iend-ist+1)/32;
    else
      num_workers2 = 32;

#ifndef CRPL_COMP
#elif CRPL_COMP == 0
    #pragma omp target teams 
    #pragma omp distribute // private(k, i, m) 
#endif
    for (k = 1; k < nz - 1; k++) {
      for (i = ist; i <= iend; i++) {
        for (m = 0; m < 5; m++) {
          rsd[m][k][1][i] = rsd[m][k][1][i]
                                         - dssp * ( + 5.0 * u[m][k][1][i]
                                                                       - 4.0 * u[m][k][2][i]
                                                                                          +       u[m][k][3][i] );
          rsd[m][k][2][i] = rsd[m][k][2][i]
                                         - dssp * ( - 4.0 * u[m][k][1][i]
                                                                       + 6.0 * u[m][k][2][i]
                                                                                          - 4.0 * u[m][k][3][i]
                                                                                                             +       u[m][k][4][i] );
        }
      }
    }

    unsigned int num_workers4 = 0;
    if((ny-6)<8)
      num_workers4 = ny-6;
    else
      num_workers4 = 8;

#ifndef CRPL_COMP
#elif CRPL_COMP == 0
   #pragma omp target teams map (alloc: rsd, u) \
            num_teams(jend-jst+1)
    #pragma omp distribute parallel for collapse(3) private (m, i)
#endif
    for (k = 1; k < nz - 1; k++) {
      for (j = 3; j < ny - 3; j++) {
        for (i = ist; i <= iend; i++) {
          for (m = 0; m < 5; m++) {
            rsd[m][k][j][i] = rsd[m][k][j][i]
                                           - dssp * (         u[m][k][j-2][i]
                                                                           - 4.0 * u[m][k][j-1][i]
                                                                                                + 6.0 * u[m][k][j][i]
                                                                                                                   - 4.0 * u[m][k][j+1][i]
                                                                                                                                        +       u[m][k][j+2][i] );
          }
        }
      }
    }

#ifndef CRPL_COMP 
#elif CRPL_COMP == 0
    #pragma omp target teams map (alloc: rsd, u) \
    num_teams(nz - 2)
    #pragma omp distribute parallel for collapse(2) private (k, i, m)
#endif
    for (k = 1; k < nz - 1; k++) {
      for (i = ist; i <= iend; i++) {
        for (m = 0; m < 5; m++) {
          rsd[m][k][ny-3][i] = rsd[m][k][ny-3][i]
                                               - dssp * (         u[m][k][ny-5][i]
                                                                                - 4.0 * u[m][k][ny-4][i]
                                                                                                      + 6.0 * u[m][k][ny-3][i]
                                                                                                                            - 4.0 * u[m][k][ny-2][i] );
          rsd[m][k][ny-2][i] = rsd[m][k][ny-2][i]
                                               - dssp * (         u[m][k][ny-4][i]
                                                                                - 4.0 * u[m][k][ny-3][i]
                                                                                                      + 5.0 * u[m][k][ny-2][i] );
        }
      }

    }
    
    if (timeron) timer_stop(t_rhsy);

    if (timeron) timer_start(t_rhsz);
    //---------------------------------------------------------------------
    // zeta-direction flux differences
    //---------------------------------------------------------------------
#ifndef CRPL_COMP
#elif CRPL_COMP == 0
    #pragma omp target teams map (alloc: utmp_G, u) \
    num_teams(jend-jst)
    #pragma omp distribute parallel for collapse(3) private(j, i, k)
#endif
    for (j = jst; j <= jend; j++) {
      for (i = ist; i <= iend; i++) {
        for (k = 0; k < nz; k++) {
          utmp_G[0][j][i][k] = u[0][k][j][i];
          utmp_G[1][j][i][k] = u[1][k][j][i];
          utmp_G[2][j][i][k] = u[2][k][j][i];
          utmp_G[3][j][i][k] = u[3][k][j][i];
          utmp_G[4][j][i][k] = u[4][k][j][i];
          utmp_G[5][j][i][k] = rho_i[k][j][i];
        }
      }
    }

#ifndef CRPL_COMP
#elif CRPL_COMP == 0
    #pragma omp target teams map(alloc: flux_G, utmp_G, qs)\
    num_teams(jend-jst)
    #pragma omp distribute parallel for collapse(3) private(j, i, k, u41, q)
#endif
    for (j = jst; j <= jend; j++) {
      for (i = ist; i <= iend; i++) {
        for (k = 0; k < nz; k++) {
          flux_G[0][j][i][k] = utmp_G[3][j][i][k];
          u41 = utmp_G[3][j][i][k] * utmp_G[5][j][i][k];

          q = qs[k][j][i];

          flux_G[1][j][i][k] = utmp_G[1][j][i][k] * u41;
          flux_G[2][j][i][k] = utmp_G[2][j][i][k] * u41;
          flux_G[3][j][i][k] = utmp_G[3][j][i][k] * u41 + C2 * (utmp_G[4][j][i][k]-q);
          flux_G[4][j][i][k] = ( C1 * utmp_G[4][j][i][k] - C2 * q ) * u41;
        }
      }
    }

#ifndef CRPL_COMP
#elif CRPL_COMP == 0
    #pragma omp target teams map (alloc: rsd, rtmp_G, flux_G) \
            num_teams(jend-jst+1)
    #pragma omp distribute parallel for collapse(3)
#endif
    for (j = jst; j <= jend; j++) {
      for (i = ist; i <= iend; i++) {
        for (k = 1; k < nz - 1; k++) {
          for (m = 0; m < 5; m++) {
            rtmp_G[m][j][i][k] =  rsd[m][k][j][i]
                                               - tz2 * ( flux_G[m][j][i][k+1] - flux_G[m][j][i][k-1] );
          }
        }
      }
    }

#ifndef CRPL_COMP
    #pragma omp target teams 
    #pragma omp distribute // \
            private(j, i, k, u21k, u31k, u41k, u51k, tmp, u21km1, u31km1, u41km1, u51km1)
#elif CRPL_COMP == 0
    #pragma omp target teams map (alloc: utmp_G, flux_G) \
    num_teams(jend - jst)
    #pragma omp distribute parallel for collapse(3)\
            private(j, i, k, u21k, u31k, u41k, u51k, tmp, u21km1, u31km1, u41km1, u51km1)
#endif
    for (j = jst; j <= jend; j++) {
      for (i = ist; i <= iend; i++) {
        for (k = 1; k < nz; k++) {
          tmp = utmp_G[5][j][i][k];

          u21k = tmp * utmp_G[1][j][i][k];
          u31k = tmp * utmp_G[2][j][i][k];
          u41k = tmp * utmp_G[3][j][i][k];
          u51k = tmp * utmp_G[4][j][i][k];

          tmp = utmp_G[5][j][i][k-1];

          u21km1 = tmp * utmp_G[1][j][i][k-1];
          u31km1 = tmp * utmp_G[2][j][i][k-1];
          u41km1 = tmp * utmp_G[3][j][i][k-1];
          u51km1 = tmp * utmp_G[4][j][i][k-1];

          flux_G[1][j][i][k] = tz3 * ( u21k - u21km1 );
          flux_G[2][j][i][k] = tz3 * ( u31k - u31km1 );
          flux_G[3][j][i][k] = (4.0/3.0) * tz3 * (u41k-u41km1);
          flux_G[4][j][i][k] = 0.50 * ( 1.0 - C1*C5 )
              * tz3 * ( ( u21k*u21k     + u31k*u31k     + u41k*u41k )
                  - ( u21km1*u21km1 + u31km1*u31km1 + u41km1*u41km1 ) )
                  + (1.0/6.0)
                  * tz3 * ( u41k*u41k - u41km1*u41km1 )
                  + C1 * C5 * tz3 * ( u51k - u51km1 );
        }
      }
    }

#ifndef CRPL_COMP
#elif CRPL_COMP == 0
    #pragma omp target map(alloc: rtmp_G, utmp_G, flux_G) 
    #pragma omp teams distribute parallel for collapse(3) private(i,j,k) num_teams(jend - jst)
#endif
    for (j = jst; j <= jend; j++) {
      for (i = ist; i <= iend; i++) {
        for (k = 1; k < nz - 1; k++) {
          rtmp_G[0][j][i][k] = rtmp_G[0][j][i][k]
                                               + dz1 * tz1 * (         utmp_G[0][j][i][k-1]
                                                                                       - 2.0 * utmp_G[0][j][i][k]
                                                                                                               +       utmp_G[0][j][i][k+1] );
          rtmp_G[1][j][i][k] = rtmp_G[1][j][i][k]
                                               + tz3 * C3 * C4 * ( flux_G[1][j][i][k+1] - flux_G[1][j][i][k] )
                                               + dz2 * tz1 * (         utmp_G[1][j][i][k-1]
                                                                                       - 2.0 * utmp_G[1][j][i][k]
                                                                                                               +       utmp_G[1][j][i][k+1] );
          rtmp_G[2][j][i][k] = rtmp_G[2][j][i][k]
                                               + tz3 * C3 * C4 * ( flux_G[2][j][i][k+1] - flux_G[2][j][i][k] )
                                               + dz3 * tz1 * (         utmp_G[2][j][i][k-1]
                                                                                       - 2.0 * utmp_G[2][j][i][k]
                                                                                                               +       utmp_G[2][j][i][k+1] );
          rtmp_G[3][j][i][k] = rtmp_G[3][j][i][k]
                                               + tz3 * C3 * C4 * ( flux_G[3][j][i][k+1] - flux_G[3][j][i][k] )
                                               + dz4 * tz1 * (         utmp_G[3][j][i][k-1]
                                                                                       - 2.0 * utmp_G[3][j][i][k]
                                                                                                               +       utmp_G[3][j][i][k+1] );
          rtmp_G[4][j][i][k] = rtmp_G[4][j][i][k]
                                               + tz3 * C3 * C4 * ( flux_G[4][j][i][k+1] - flux_G[4][j][i][k] )
                                               + dz5 * tz1 * (         utmp_G[4][j][i][k-1]
                                                                                       - 2.0 * utmp_G[4][j][i][k]
                                                                                                               +       utmp_G[4][j][i][k+1] );
        }
      }
    }

    //---------------------------------------------------------------------
    // fourth-order dissipation
    //---------------------------------------------------------------------
#ifndef CRPL_COMP
    #pragma omp target teams 
    #pragma omp distribute // private(j, i, m)
#elif CRPL_COMP == 0
    #pragma omp target teams 
    #pragma omp distribute // private(j, i, m)
#endif
    for (j = jst; j <= jend; j++) {
      for (i = ist; i <= iend; i++) {
        for (m = 0; m < 5; m++) {
          rsd[m][1][j][i] = rtmp_G[m][j][i][1]
                                            - dssp * ( + 5.0 * utmp_G[m][j][i][1]
                                                                               - 4.0 * utmp_G[m][j][i][2]
                                                                                                       +       utmp_G[m][j][i][3] );
          rsd[m][2][j][i] = rtmp_G[m][j][i][2]
                                            - dssp * ( - 4.0 * utmp_G[m][j][i][1]
                                                                               + 6.0 * utmp_G[m][j][i][2]
                                                                                                       - 4.0 * utmp_G[m][j][i][3]
                                                                                                                               +       utmp_G[m][j][i][4] );
        }
      }
    }

#ifndef CRPL_COMP
#elif CRPL_COMP == 0
    #pragma omp target teams map (alloc: rsd, rtmp_G, utmp_G) \
    num_teams(jend-jst) //thread_limit(num_workers3) 
    #pragma omp distribute parallel for collapse(3) private (m, k)
#endif
    for (j = jst; j <= jend; j++) {
      for (i = ist; i <= iend; i++) {
        for (k = 3; k < nz - 3; k++) {
          for (m = 0; m < 5; m++) {
            rsd[m][k][j][i] = rtmp_G[m][j][i][k]
                                              - dssp * (         utmp_G[m][j][i][k-2]
                                                                                 - 4.0 * utmp_G[m][j][i][k-1]
                                                                                                         + 6.0 * utmp_G[m][j][i][k]
                                                                                                                                 - 4.0 * utmp_G[m][j][i][k+1]
                                                                                                                                                         +       utmp_G[m][j][i][k+2] );
          }
        }
      }
    }

#ifndef CRPL_COMP
#elif CRPL_COMP == 0
    #pragma omp target teams map (alloc: rsd, rtmp_G, utmp_G) \
    num_teams(jend - jst)
    #pragma omp distribute parallel for collapse(2) private (j, i, m)
#endif
    for (j = jst; j <= jend; j++) {
      for (i = ist; i <= iend; i++) {
        for (m = 0; m < 5; m++) {
          rsd[m][nz-3][j][i] = rtmp_G[m][j][i][nz-3]
                                               - dssp * (         utmp_G[m][j][i][nz-5]
                                                                                  - 4.0 * utmp_G[m][j][i][nz-4]
                                                                                                          + 6.0 * utmp_G[m][j][i][nz-3]
                                                                                                                                  - 4.0 * utmp_G[m][j][i][nz-2] );
          rsd[m][nz-2][j][i] = rtmp_G[m][j][i][nz-2]
                                               - dssp * (         utmp_G[m][j][i][nz-4]
                                                                                  - 4.0 * utmp_G[m][j][i][nz-3]
                                                                                                          + 5.0 * utmp_G[m][j][i][nz-2] );
        }
      }
    }
} // End of omp target data    
    if (timeron) timer_stop(t_rhsz);
    if (timeron) timer_stop(t_rhs);

}


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
#include <stdio.h>

//---------------------------------------------------------------------
//
// compute the right hand side based on exact solution
//
//---------------------------------------------------------------------
void erhs()
{
  //---------------------------------------------------------------------
  // local variables
  //---------------------------------------------------------------------
  int i, j, k, m;
  double xi, eta, zeta;
  double q;
  double u21, u31, u41;
  double tmp;
  double u21i, u31i, u41i, u51i;
  double u21j, u31j, u41j, u51j;
  double u21k, u31k, u41k, u51k;
  double u21im1, u31im1, u41im1, u51im1;
  double u21jm1, u31jm1, u41jm1, u51jm1;
  double u21km1, u31km1, u41km1, u51km1;
  //unsigned int num_workers = 0;

#pragma omp target data map(alloc: frct, rsd, ce, flux_G)
  {  
#ifndef CRPL_COMP
    #pragma omp target teams 
    #pragma omp distribute // private(k, j, i, m)
#elif CRPL_COMP == 0
    #pragma omp target teams
    {
      #pragma omp distribute // for private(k, j, i, m)
#endif
      for (k = 0; k < nz; k++) {
        for (j = 0; j < ny; j++) {
          for (i = 0; i < nx; i++) {
            for (m = 0; m < 5; m++) {
              frct[m][k][j][i] = 0.0;
            }
          }
        }
      }
    }

#ifndef CRPL_COMP
    #pragma omp target teams 
    #pragma omp distribute // private(k, j, i, m, zeta, eta, xi)
#elif CRPL_COMP == 0
    #pragma omp target teams 
    #pragma omp distribute // private(k, j, i, m, zeta, eta, xi)
#endif
    for (k = 0; k < nz; k++) {
      zeta = ( (double)k ) / ( nz - 1 );
      for (j = 0; j < ny; j++) {
        eta = ( (double)j ) / ( ny0 - 1 );
        for (i = 0; i < nx; i++) {
          xi = ( (double)i ) / ( nx0 - 1 );
          for (m = 0; m < 5; m++) {
            rsd[m][k][j][i] =  ce[m][0]
                                     + (ce[m][1]
                                              + (ce[m][4]
                                                       + (ce[m][7]
                                                                +  ce[m][10] * xi) * xi) * xi) * xi
                                                                + (ce[m][2]
                                                                         + (ce[m][5]
                                                                                  + (ce[m][8]
                                                                                           +  ce[m][11] * eta) * eta) * eta) * eta
                                                                                           + (ce[m][3]
                                                                                                    + (ce[m][6]
                                                                                                             + (ce[m][9]
                                                                                                                      +  ce[m][12] * zeta) * zeta) * zeta) * zeta;
          }
        }
      }
    }

    //---------------------------------------------------------------------
    // xi-direction flux differences
    //---------------------------------------------------------------------
#ifndef CRPL_COMP
    #pragma omp target teams 
    #pragma omp distribute // private(k, j, i, u21, q)
#elif CRPL_COMP == 0
    #pragma omp target teams 
    #pragma omp distribute // private(k, j, i, u21, q)
#endif
    for (k = 1; k < nz - 1; k++) {
      for (j = jst; j <= jend; j++) {
        for (i = 0; i < nx; i++) {
          flux_G[0][k][j][i] = rsd[1][k][j][i];
          u21 = rsd[1][k][j][i] / rsd[0][k][j][i];
          q = 0.50 * (  rsd[1][k][j][i] * rsd[1][k][j][i]
                                                       + rsd[2][k][j][i] * rsd[2][k][j][i]
                                                                                        + rsd[3][k][j][i] * rsd[3][k][j][i] )
                     / rsd[0][k][j][i];
          flux_G[1][k][j][i] = rsd[1][k][j][i] * u21 + C2 * ( rsd[4][k][j][i] - q );
          flux_G[2][k][j][i] = rsd[2][k][j][i] * u21;
          flux_G[3][k][j][i] = rsd[3][k][j][i] * u21;
          flux_G[4][k][j][i] = ( C1 * rsd[4][k][j][i] - C2 * q ) * u21;
        }
      }
    }

#ifndef CRPL_COMP
    #pragma omp target teams 
    #pragma omp distribute // private(k, j, i, m)
#elif CRPL_COMP == 0
    #pragma omp target teams
    #pragma omp distribute// for private(k, j, i, m)
#endif
    for (k = 1; k < nz - 1; k++) {
      for (j = jst; j <= jend; j++) {
        for (i = ist; i <= iend; i++) {
          for (m = 0; m < 5; m++) {
            frct[m][k][j][i] =  frct[m][k][j][i]
                                              - tx2 * ( flux_G[m][k][j][i+1] - flux_G[m][k][j][i-1] );
          }
        }
      }
    }

#ifndef CRPL_COMP
    #pragma omp target teams 
    #pragma omp distribute // \
            private(k, j, i, m, tmp, u21i, u31i, u41i, u51i, u21im1, u31im1, u41im1, u51im1)
#elif CRPL_COMP == 0
    #pragma omp target teams 
    #pragma omp distribute // \
            private(k, j, i, m, tmp, u21i, u31i, u41i, u51i, u21im1, u31im1, u41im1, u51im1)
#endif
    for (k = 1; k < nz - 1; k++) {
      for (j = jst; j <= jend; j++) {
        for (i = ist; i < nx; i++) {
          tmp = 1.0 / rsd[0][k][j][i];

          u21i = tmp * rsd[1][k][j][i];
          u31i = tmp * rsd[2][k][j][i];
          u41i = tmp * rsd[3][k][j][i];
          u51i = tmp * rsd[4][k][j][i];

          tmp = 1.0 / rsd[0][k][j][i-1];

          u21im1 = tmp * rsd[1][k][j][i-1];
          u31im1 = tmp * rsd[2][k][j][i-1];
          u41im1 = tmp * rsd[3][k][j][i-1];
          u51im1 = tmp * rsd[4][k][j][i-1];

          flux_G[1][k][j][i] = (4.0/3.0) * tx3 * ( u21i - u21im1 );
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
    #pragma omp target teams 
    #pragma omp distribute // private (k, j, i)
#elif CRPL_COMP == 0
    #pragma omp target teams 
    #pragma omp distribute // private (k, j, i)
#endif
    for (k = 1; k < nz - 1; k++) {
      for (j = jst; j <= jend; j++) {
        for (i = ist; i <= iend; i++) {
          frct[0][k][j][i] = frct[0][k][j][i]
                                           + dx1 * tx1 * (        rsd[0][k][j][i-1]
                                                                               - 2.0 * rsd[0][k][j][i]
                                                                                                    +       rsd[0][k][j][i+1] );
          frct[1][k][j][i] = frct[1][k][j][i]
                                           + tx3 * C3 * C4 * ( flux_G[1][k][j][i+1] - flux_G[1][k][j][i] )
                                           + dx2 * tx1 * (        rsd[1][k][j][i-1]
                                                                               - 2.0 * rsd[1][k][j][i]
                                                                                                    +       rsd[1][k][j][i+1] );
          frct[2][k][j][i] = frct[2][k][j][i]
                                           + tx3 * C3 * C4 * ( flux_G[2][k][j][i+1] - flux_G[2][k][j][i] )
                                           + dx3 * tx1 * (        rsd[2][k][j][i-1]
                                                                               - 2.0 * rsd[2][k][j][i]
                                                                                                    +       rsd[2][k][j][i+1] );
          frct[3][k][j][i] = frct[3][k][j][i]
                                           + tx3 * C3 * C4 * ( flux_G[3][k][j][i+1] - flux_G[3][k][j][i] )
                                           + dx4 * tx1 * (        rsd[3][k][j][i-1]
                                                                               - 2.0 * rsd[3][k][j][i]
                                                                                                    +       rsd[3][k][j][i+1] );
          frct[4][k][j][i] = frct[4][k][j][i]
                                           + tx3 * C3 * C4 * ( flux_G[4][k][j][i+1] - flux_G[4][k][j][i] )
                                           + dx5 * tx1 * (        rsd[4][k][j][i-1]
                                                                               - 2.0 * rsd[4][k][j][i]
                                                                                                    +       rsd[4][k][j][i+1] );
        }
      }
    }

    //---------------------------------------------------------------------
    // Fourth-order dissipation
    //---------------------------------------------------------------------
    //num_workers = (jend-jst+1)/32;
#ifndef CRPL_COMP
    #pragma omp target teams 
    #pragma omp distribute // private (k, j, m)
#elif CRPL_COMP == 0
    #pragma omp target teams 
    #pragma omp distribute // private (k, j, m)
#endif
    for (k = 1; k < nz - 1; k++) {
      for (j = jst; j <= jend; j++) {
        for (m = 0; m < 5; m++) {
          frct[m][k][j][1] = frct[m][k][j][1]
                                           - dssp * ( + 5.0 * rsd[m][k][j][1]
                                                                           - 4.0 * rsd[m][k][j][2]
                                                                                                +       rsd[m][k][j][3] );
          frct[m][k][j][2] = frct[m][k][j][2]
                                           - dssp * ( - 4.0 * rsd[m][k][j][1]
                                                                           + 6.0 * rsd[m][k][j][2]
                                                                                                - 4.0 * rsd[m][k][j][3]
                                                                                                                     +       rsd[m][k][j][4] );
        }
      }
    }

#ifndef CRPL_COMP
    #pragma omp target teams 
    #pragma omp distribute // private (k, j, i, m)
#elif CRPL_COMP == 0
    #pragma omp target teams 
    #pragma omp distribute // private (k, j, i, m)
#endif
    for (k = 1; k < nz - 1; k++) {
      for (j = jst; j <= jend; j++) {
        for (i = 3; i < nx - 3; i++) {
          for (m = 0; m < 5; m++) {
            frct[m][k][j][i] = frct[m][k][j][i]
                                             - dssp * (        rsd[m][k][j][i-2]
                                                                            - 4.0 * rsd[m][k][j][i-1]
                                                                                                 + 6.0 * rsd[m][k][j][i]
                                                                                                                      - 4.0 * rsd[m][k][j][i+1]
                                                                                                                                           +       rsd[m][k][j][i+2] );
          }
        }
      }
    }

#ifndef CRPL_COMP
    #pragma omp target teams 
    #pragma omp distribute // private (k, j, m)
#elif CRPL_COMP == 0
    #pragma omp target teams 
    #pragma omp distribute // private (k, j, m)
#endif
    for (k = 1; k < nz - 1; k++) {
      for (j = jst; j <= jend; j++) {
        for (m = 0; m < 5; m++) {
          frct[m][k][j][nx-3] = frct[m][k][j][nx-3]
                                              - dssp * (        rsd[m][k][j][nx-5]
                                                                             - 4.0 * rsd[m][k][j][nx-4]
                                                                                                  + 6.0 * rsd[m][k][j][nx-3]
                                                                                                                       - 4.0 * rsd[m][k][j][nx-2] );
          frct[m][k][j][nx-2] = frct[m][k][j][nx-2]
                                              - dssp * (        rsd[m][k][j][nx-4]
                                                                             - 4.0 * rsd[m][k][j][nx-3]
                                                                                                  + 5.0 * rsd[m][k][j][nx-2] );
        }
      }
    }

    //---------------------------------------------------------------------
    // eta-direction flux differences
    //---------------------------------------------------------------------
#ifndef CRPL_COMP
    #pragma omp target teams 
    #pragma omp distribute // private (k, j, i, u31, q)
#elif CRPL_COMP == 0
    #pragma omp target teams 
    #pragma omp distribute // private (k, j, i, u31, q)
#endif
    for (k = 1; k < nz - 1; k++) {
      for (i = ist; i <= iend; i++) {
        for (j = 0; j < ny; j++) {
          flux_G[0][k][i][j] = rsd[2][k][j][i];
          u31 = rsd[2][k][j][i] / rsd[0][k][j][i];
          q = 0.50 * (  rsd[1][k][j][i] * rsd[1][k][j][i]
                                                       + rsd[2][k][j][i] * rsd[2][k][j][i]
                                                                                        + rsd[3][k][j][i] * rsd[3][k][j][i] )
                     / rsd[0][k][j][i];
          flux_G[1][k][i][j] = rsd[1][k][j][i] * u31;
          flux_G[2][k][i][j] = rsd[2][k][j][i] * u31 + C2 * ( rsd[4][k][j][i] - q );
          flux_G[3][k][i][j] = rsd[3][k][j][i] * u31;
          flux_G[4][k][i][j] = ( C1 * rsd[4][k][j][i] - C2 * q ) * u31;
        }
      }
    }

#ifndef CRPL_COMP
    #pragma omp target teams 
    #pragma omp distribute // private (k, j, i, m)
#elif CRPL_COMP == 0
    #pragma omp target teams 
    #pragma omp distribute // private (k, j, i, m)
#endif
    for (k = 1; k < nz - 1; k++) {
      for (i = ist; i <= iend; i++) {
        for (j = jst; j <= jend; j++) {
          for (m = 0; m < 5; m++) {
            frct[m][k][j][i] =  frct[m][k][j][i]
                                              - ty2 * ( flux_G[m][k][i][j+1] - flux_G[m][k][i][j-1] );
          }
        }
      }
    }

#ifndef CRPL_COMP
    #pragma omp target teams 
    #pragma omp distribute // \
            private (k, j, i, tmp, u21j, u31j, u41j, u51j, u21jm1, u31jm1, u41jm1, u51jm1)
#elif CRPL_COMP == 0
    #pragma omp target teams 
    #pragma omp distribute // \
            private (k, j, i, tmp, u21j, u31j, u41j, u51j, u21jm1, u31jm1, u41jm1, u51jm1)
#endif
    for (k = 1; k < nz - 1; k++) {
      for (i = ist; i <= iend; i++) {
        for (j = jst; j < ny; j++) {
          tmp = 1.0 / rsd[0][k][j][i];

          u21j = tmp * rsd[1][k][j][i];
          u31j = tmp * rsd[2][k][j][i];
          u41j = tmp * rsd[3][k][j][i];
          u51j = tmp * rsd[4][k][j][i];

          tmp = 1.0 / rsd[0][k][j-1][i];

          u21jm1 = tmp * rsd[1][k][j-1][i];
          u31jm1 = tmp * rsd[2][k][j-1][i];
          u41jm1 = tmp * rsd[3][k][j-1][i];
          u51jm1 = tmp * rsd[4][k][j-1][i];

          flux_G[1][k][i][j] = ty3 * ( u21j - u21jm1 );
          flux_G[2][k][i][j] = (4.0/3.0) * ty3 * ( u31j - u31jm1 );
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
    #pragma omp target teams 
    #pragma omp distribute // private (k, j, i)
#elif CRPL_COMP == 0
    #pragma omp target teams 
    #pragma omp distribute // private (k, j, i)
#endif
    for (k = 1; k < nz - 1; k++) {
      for (i = ist; i <= iend; i++) {
        for (j = jst; j <= jend; j++) {
          frct[0][k][j][i] = frct[0][k][j][i]
                                           + dy1 * ty1 * (        rsd[0][k][j-1][i]
                                                                                 - 2.0 * rsd[0][k][j][i]
                                                                                                      +       rsd[0][k][j+1][i] );
          frct[1][k][j][i] = frct[1][k][j][i]
                                           + ty3 * C3 * C4 * ( flux_G[1][k][i][j+1] - flux_G[1][k][i][j] )
                                           + dy2 * ty1 * (        rsd[1][k][j-1][i]
                                                                                 - 2.0 * rsd[1][k][j][i]
                                                                                                      +       rsd[1][k][j+1][i] );
          frct[2][k][j][i] = frct[2][k][j][i]
                                           + ty3 * C3 * C4 * ( flux_G[2][k][i][j+1] - flux_G[2][k][i][j] )
                                           + dy3 * ty1 * (        rsd[2][k][j-1][i]
                                                                                 - 2.0 * rsd[2][k][j][i]
                                                                                                      +       rsd[2][k][j+1][i] );
          frct[3][k][j][i] = frct[3][k][j][i]
                                           + ty3 * C3 * C4 * ( flux_G[3][k][i][j+1] - flux_G[3][k][i][j] )
                                           + dy4 * ty1 * (        rsd[3][k][j-1][i]
                                                                                 - 2.0 * rsd[3][k][j][i]
                                                                                                      +       rsd[3][k][j+1][i] );
          frct[4][k][j][i] = frct[4][k][j][i]
                                           + ty3 * C3 * C4 * ( flux_G[4][k][i][j+1] - flux_G[4][k][i][j] )
                                           + dy5 * ty1 * (        rsd[4][k][j-1][i]
                                                                                 - 2.0 * rsd[4][k][j][i]
                                                                                                      +       rsd[4][k][j+1][i] );
        }
      }
    }

    //---------------------------------------------------------------------
    // fourth-order dissipation
    //---------------------------------------------------------------------

    //num_workers = (iend-ist+1)/32;
#ifndef CRPL_COMP
    #pragma omp target teams 
    #pragma omp distribute // private (k, i, m)
#elif CRPL_COMP == 0
    #pragma omp target teams 
    #pragma omp distribute // private (k, i, m)
#endif
    for (k = 1; k < nz - 1; k++) {
      for (i = ist; i <= iend; i++) {
        for (m = 0; m < 5; m++) {
          frct[m][k][1][i] = frct[m][k][1][i]
                                           - dssp * ( + 5.0 * rsd[m][k][1][i]
                                                                           - 4.0 * rsd[m][k][2][i]
                                                                                                +       rsd[m][k][3][i] );
          frct[m][k][2][i] = frct[m][k][2][i]
                                           - dssp * ( - 4.0 * rsd[m][k][1][i]
                                                                           + 6.0 * rsd[m][k][2][i]
                                                                                                - 4.0 * rsd[m][k][3][i]
                                                                                                                     +       rsd[m][k][4][i] );
        }
      }
    }

#ifndef CRPL_COMP
    #pragma omp target teams 
    #pragma omp distribute // private (k, i, j, m)
#elif CRPL_COMP == 0
    #pragma omp target teams 
    #pragma omp distribute // private (k, i, j, m) 
#endif
    for (k = 1; k < nz - 1; k++) {
      for (i = ist; i <= iend; i++) {
        for (j = 3; j < ny - 3; j++) {
          for (m = 0; m < 5; m++) {
            frct[m][k][j][i] = frct[m][k][j][i]
                                             - dssp * (        rsd[m][k][j-2][i]
                                                                              - 4.0 * rsd[m][k][j-1][i]
                                                                                                     + 6.0 * rsd[m][k][j][i]
                                                                                                                          - 4.0 * rsd[m][k][j+1][i]
                                                                                                                                                 +       rsd[m][k][j+2][i] );
          }
        }
      }
    }

#ifndef CRPL_COMP
    #pragma omp target teams 
    #pragma omp distribute // private (k, i, m)
#elif CRPL_COMP == 0
    #pragma omp target teams 
    #pragma omp distribute // private (k, i, m)
#endif
    for (k = 1; k < nz - 1; k++) {
      for (i = ist; i <= iend; i++) {
        for (m = 0; m < 5; m++) {
          frct[m][k][ny-3][i] = frct[m][k][ny-3][i]
                                                 - dssp * (        rsd[m][k][ny-5][i]
                                                                                   - 4.0 * rsd[m][k][ny-4][i]
                                                                                                           + 6.0 * rsd[m][k][ny-3][i]
                                                                                                                                   - 4.0 * rsd[m][k][ny-2][i] );
          frct[m][k][ny-2][i] = frct[m][k][ny-2][i]
                                                 - dssp * (        rsd[m][k][ny-4][i]
                                                                                   - 4.0 * rsd[m][k][ny-3][i]
                                                                                                           + 5.0 * rsd[m][k][ny-2][i] );
        }
      }
    }

    //---------------------------------------------------------------------
    // zeta-direction flux differences
    //---------------------------------------------------------------------
#ifndef CRPL_COMP
    #pragma omp target teams 
    #pragma omp distribute // private (k, i, j, u41, q)
#elif CRPL_COMP == 0
    #pragma omp target teams 
    #pragma omp distribute // private (k, i, j, u41, q)
#endif
    for (j = jst; j <= jend; j++) {
      for (i = ist; i <= iend; i++) {
        for (k = 0; k < nz; k++) {
          flux_G[0][j][i][k] = rsd[3][k][j][i];
          u41 = rsd[3][k][j][i] / rsd[0][k][j][i];
          q = 0.50 * (  rsd[1][k][j][i] * rsd[1][k][j][i]
                                                       + rsd[2][k][j][i] * rsd[2][k][j][i]
                                                                                        + rsd[3][k][j][i] * rsd[3][k][j][i] )
                     / rsd[0][k][j][i];
          flux_G[1][j][i][k] = rsd[1][k][j][i] * u41;
          flux_G[2][j][i][k] = rsd[2][k][j][i] * u41;
          flux_G[3][j][i][k] = rsd[3][k][j][i] * u41 + C2 * ( rsd[4][k][j][i] - q );
          flux_G[4][j][i][k] = ( C1 * rsd[4][k][j][i] - C2 * q ) * u41;
        }
      }
    }

#ifndef CRPL_COMP
    #pragma omp target teams 
    #pragma omp distribute // private (k, i, j, m)
#elif CRPL_COMP == 0
    #pragma omp target teams 
    #pragma omp distribute // private (k, i, j, m)
#endif
    for (j = jst; j <= jend; j++) {
      for (i = ist; i <= iend; i++) {
        for (k = 1; k < nz - 1; k++) {
          for (m = 0; m < 5; m++) {
            frct[m][k][j][i] =  frct[m][k][j][i]
                                              - tz2 * ( flux_G[m][j][i][k+1] - flux_G[m][j][i][k-1] );
          }
        }
      }
    }

#ifndef CRPL_COMP
    #pragma omp target teams 
    #pragma omp distribute // \
            private (j, i, k, tmp, u21k, u31k, u41k, u51k, u21km1, u31km1, u41km1, u51km1 )
#elif CRPL_COMP == 0
    #pragma omp target teams 
    #pragma omp distribute // \
            private (j, i, k, tmp, u21k, u31k, u41k, u51k, u21km1, u31km1, u41km1, u51km1 )
#endif
    for (j = jst; j <= jend; j++) {
      for (i = ist; i <= iend; i++) {
        for (k = 1; k < nz; k++) {
          tmp = 1.0 / rsd[0][k][j][i];

          u21k = tmp * rsd[1][k][j][i];
          u31k = tmp * rsd[2][k][j][i];
          u41k = tmp * rsd[3][k][j][i];
          u51k = tmp * rsd[4][k][j][i];

          tmp = 1.0 / rsd[0][k-1][j][i];

          u21km1 = tmp * rsd[1][k-1][j][i];
          u31km1 = tmp * rsd[2][k-1][j][i];
          u41km1 = tmp * rsd[3][k-1][j][i];
          u51km1 = tmp * rsd[4][k-1][j][i];

          flux_G[1][j][i][k] = tz3 * ( u21k - u21km1 );
          flux_G[2][j][i][k] = tz3 * ( u31k - u31km1 );
          flux_G[3][j][i][k] = (4.0/3.0) * tz3 * ( u41k - u41km1 );
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
    #pragma omp target teams 
    #pragma omp distribute // private(j, i, k)
#elif CRPL_COMP == 0
    #pragma omp target teams 
    #pragma omp distribute // private(j, i, k)
#endif
    for (j = jst; j <= jend; j++) {
      for (i = ist; i <= iend; i++) {
        for (k = 1; k < nz - 1; k++) {
          frct[0][k][j][i] = frct[0][k][j][i]
                                           + dz1 * tz1 * (        rsd[0][k+1][j][i]
                                                                                 - 2.0 * rsd[0][k][j][i]
                                                                                                      +       rsd[0][k-1][j][i] );
          frct[1][k][j][i] = frct[1][k][j][i]
                                           + tz3 * C3 * C4 * ( flux_G[1][j][i][k+1] - flux_G[1][j][i][k] )
                                           + dz2 * tz1 * (        rsd[1][k+1][j][i]
                                                                                 - 2.0 * rsd[1][k][j][i]
                                                                                                      +       rsd[1][k-1][j][i] );
          frct[2][k][j][i] = frct[2][k][j][i]
                                           + tz3 * C3 * C4 * ( flux_G[2][j][i][k+1] - flux_G[2][j][i][k] )
                                           + dz3 * tz1 * (        rsd[2][k+1][j][i]
                                                                                 - 2.0 * rsd[2][k][j][i]
                                                                                                      +       rsd[2][k-1][j][i] );
          frct[3][k][j][i] = frct[3][k][j][i]
                                           + tz3 * C3 * C4 * ( flux_G[3][j][i][k+1] - flux_G[3][j][i][k] )
                                           + dz4 * tz1 * (        rsd[3][k+1][j][i]
                                                                                 - 2.0 * rsd[3][k][j][i]
                                                                                                      +       rsd[3][k-1][j][i] );
          frct[4][k][j][i] = frct[4][k][j][i]
                                           + tz3 * C3 * C4 * ( flux_G[4][j][i][k+1] - flux_G[4][j][i][k] )
                                           + dz5 * tz1 * (        rsd[4][k+1][j][i]
                                                                                 - 2.0 * rsd[4][k][j][i]
                                                                                                      +       rsd[4][k-1][j][i] );
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
          frct[m][1][j][i] = frct[m][1][j][i]
                                           - dssp * ( + 5.0 * rsd[m][1][j][i]
                                                                           - 4.0 * rsd[m][2][j][i]
                                                                                                +       rsd[m][3][j][i] );
          frct[m][2][j][i] = frct[m][2][j][i]
                                           - dssp * ( - 4.0 * rsd[m][1][j][i]
                                                                           + 6.0 * rsd[m][2][j][i]
                                                                                                - 4.0 * rsd[m][3][j][i]
                                                                                                                     +       rsd[m][4][j][i] );
        }
      }
    }

#ifndef CRPL_COMP
    #pragma omp target teams 
    #pragma omp distribute // private(j, i, k)
#elif CRPL_COMP == 0
    #pragma omp target teams 
    #pragma omp distribute // private(j, i, k)
#endif
    for (j = jst; j <= jend; j++) {
      for (i = ist; i <= iend; i++) {
        for (k = 3; k < nz - 3; k++) {
          for (m = 0; m < 5; m++) {
            frct[m][k][j][i] = frct[m][k][j][i]
                                             - dssp * (        rsd[m][k-2][j][i]
                                                                              - 4.0 * rsd[m][k-1][j][i]
                                                                                                     + 6.0 * rsd[m][k][j][i]
                                                                                                                          - 4.0 * rsd[m][k+1][j][i]
                                                                                                                                                 +       rsd[m][k+2][j][i] );
          }
        }
      }
    }

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
          frct[m][nz-3][j][i] = frct[m][nz-3][j][i]
                                                 - dssp * (        rsd[m][nz-5][j][i]
                                                                                   - 4.0 * rsd[m][nz-4][j][i]
                                                                                                           + 6.0 * rsd[m][nz-3][j][i]
                                                                                                                                   - 4.0 * rsd[m][nz-2][j][i] );
          frct[m][nz-2][j][i] = frct[m][nz-2][j][i]
                                                 - dssp * (        rsd[m][nz-4][j][i]
                                                                                   - 4.0 * rsd[m][nz-3][j][i]
                                                                                                           + 5.0 * rsd[m][nz-2][j][i] );
        }
      }
    }
  } // End of omp target data 
}



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

#include <stdio.h>
#include "applu.incl"

void pintgr()
{
  //---------------------------------------------------------------------
  // local variables
  //---------------------------------------------------------------------
  int i, j, k;
  int ibeg, ifin, ifin1;
  int jbeg, jfin, jfin1;
  double phi1[ISIZ3+2][ISIZ2+2];
  double phi2[ISIZ3+2][ISIZ2+2];
  double frc1, frc2, frc3;

  //---------------------------------------------------------------------
  // set up the sub-domains for integeration in each processor
  //---------------------------------------------------------------------
  ibeg = ii1;
  ifin = ii2;
  jbeg = ji1;
  jfin = ji2;
  ifin1 = ifin - 1;
  jfin1 = jfin - 1;

  //---------------------------------------------------------------------
  // initialize
  //---------------------------------------------------------------------
  for (k = 0; k <= ISIZ3+1; k++) {
    for (i = 0; i <= ISIZ2+1; i++) {
      phi1[k][i] = 0.0;
      phi2[k][i] = 0.0;
    }
  }

  for (j = jbeg; j < jfin; j++) {
    for (i = ibeg; i < ifin; i++) {
      k = ki1;

      phi1[j][i] = C2*(  u[4][k][j][i]
          - 0.50 * (  u[1][k][j][i] * u[1][k][j][i]
                    + u[2][k][j][i] * u[2][k][j][i]
                    + u[3][k][j][i] * u[3][k][j][i] )
                   / u[0][k][j][i] );

      k = ki2 - 1;

      phi2[j][i] = C2*(  u[4][k][j][i]
          - 0.50 * (  u[1][k][j][i] * u[1][k][j][i]
                    + u[2][k][j][i] * u[2][k][j][i]
                    + u[3][k][j][i] * u[3][k][j][i] )
                   / u[0][k][j][i] );
    }
  }

  frc1 = 0.0;
  for (j = jbeg; j < jfin1; j++) {
    for (i = ibeg; i < ifin1; i++) {
      frc1 = frc1 + (  phi1[j][i]
                     + phi1[j][i+1]
                     + phi1[j+1][i]
                     + phi1[j+1][i+1]
                     + phi2[j][i]
                     + phi2[j][i+1]
                     + phi2[j+1][i]
                     + phi2[j+1][i+1] );
    }
  }
  frc1 = dxi * deta * frc1;

  //---------------------------------------------------------------------
  // initialize
  //---------------------------------------------------------------------
  for (k = 0; k <= ISIZ3+1; k++) {
    for (i = 0; i <= ISIZ2+1; i++) {
      phi1[k][i] = 0.0;
      phi2[k][i] = 0.0;
    }
  }
  if (jbeg == ji1) {
    for (k = ki1; k < ki2; k++) {
      for (i = ibeg; i < ifin; i++) {
        phi1[k][i] = C2*(  u[4][k][jbeg][i]
            - 0.50 * (  u[1][k][jbeg][i] * u[1][k][jbeg][i]
                      + u[2][k][jbeg][i] * u[2][k][jbeg][i]
                      + u[3][k][jbeg][i] * u[3][k][jbeg][i] )
                     / u[0][k][jbeg][i] );
      }
    }
  }

  if (jfin == ji2) {
    for (k = ki1; k < ki2; k++) {
      for (i = ibeg; i < ifin; i++) {
        phi2[k][i] = C2*(  u[4][k][jfin-1][i]
            - 0.50 * (  u[1][k][jfin-1][i] * u[1][k][jfin-1][i]
                      + u[2][k][jfin-1][i] * u[2][k][jfin-1][i]
                      + u[3][k][jfin-1][i] * u[3][k][jfin-1][i] )
                     / u[0][k][jfin-1][i] );
      }
    }
  }

  frc2 = 0.0;
  for (k = ki1; k < ki2-1; k++) {
    for (i = ibeg; i < ifin1; i++) {
      frc2 = frc2 + (  phi1[k][i]
                     + phi1[k][i+1]
                     + phi1[k+1][i]
                     + phi1[k+1][i+1]
                     + phi2[k][i]
                     + phi2[k][i+1]
                     + phi2[k+1][i]
                     + phi2[k+1][i+1] );
    }
  }
  frc2 = dxi * dzeta * frc2;

  //---------------------------------------------------------------------
  // initialize
  //---------------------------------------------------------------------
  for (k = 0; k <= ISIZ3+1; k++) {
    for (i = 0; i <= ISIZ2+1; i++) {
      phi1[k][i] = 0.0;
      phi2[k][i] = 0.0;
    }
  }
  if (ibeg == ii1) {
    for (k = ki1; k < ki2; k++) {
      for (j = jbeg; j < jfin; j++) {
        phi1[k][j] = C2*(  u[4][k][j][ibeg]
            - 0.50 * (  u[1][k][j][ibeg] * u[1][k][j][ibeg]
                      + u[2][k][j][ibeg] * u[2][k][j][ibeg]
                      + u[3][k][j][ibeg] * u[3][k][j][ibeg] )
                     / u[0][k][j][ibeg] );
      }
    }
  }

  if (ifin == ii2) {
    for (k = ki1; k < ki2; k++) {
      for (j = jbeg; j < jfin; j++) {
        phi2[k][j] = C2*(  u[4][k][j][ifin-1]
            - 0.50 * (  u[1][k][j][ifin-1] * u[1][k][j][ifin-1]
                      + u[2][k][j][ifin-1] * u[2][k][j][ifin-1]
                      + u[3][k][j][ifin-1] * u[3][k][j][ifin-1] )
                     / u[0][k][j][ifin-1] );
      }
    }
  }

  frc3 = 0.0;
  for (k = ki1; k < ki2-1; k++) {
    for (j = jbeg; j < jfin1; j++) {
      frc3 = frc3 + (  phi1[k][j]
                     + phi1[k][j+1]
                     + phi1[k+1][j]
                     + phi1[k+1][j+1]
                     + phi2[k][j]
                     + phi2[k][j+1]
                     + phi2[k+1][j]
                     + phi2[k+1][j+1] );
    }
  }
  frc3 = deta * dzeta * frc3;

  frc = 0.25 * ( frc1 + frc2 + frc3 );
  //printf("\n\n     surface integral = %12.5E\n\n\n", frc);
}


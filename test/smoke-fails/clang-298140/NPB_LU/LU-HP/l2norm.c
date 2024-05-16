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

#include <math.h>
#include "applu.incl"

//---------------------------------------------------------------------
// to compute the l2-norm of vector v.
//---------------------------------------------------------------------
//---------------------------------------------------------------------
// To improve cache performance, second two dimensions padded by 1 
// for even number sizes only.  Only needed in v.
//---------------------------------------------------------------------
void l2norm (int ldx, int ldy, int ldz, int nx0, int ny0, int nz0,
    int ist, int iend, int jst, int jend)
{
  //---------------------------------------------------------------------
  // local variables
  //---------------------------------------------------------------------
  int i, j, k, m;
  double rsdnm0, rsdnm1, rsdnm2, rsdnm3, rsdnm4;

  rsdnm0 = (double)0.0;
  rsdnm1 = (double)0.0;
  rsdnm2 = (double)0.0;
  rsdnm3 = (double)0.0;
  rsdnm4 = (double)0.0;
#pragma omp target map (alloc: rsdnm, rsd)
  {
#ifndef CRPL_COMP
    #pragma omp parallel for private(m)
#elif CRPL_COMP == 0
    #pragma omp parallel for private(m)
#endif
    for (m = 0; m < 5; m++) {
      rsdnm[m] = 0.0;
    }
#ifndef CRPL_COMP
    #pragma omp parallel for private(k,j,i) reduction (+: rsdnm0, rsdnm1, rsdnm2, rsdnm3, rsdnm4)
#elif CRPL_COMP == 0
    #pragma omp parallel for private(k,j,i) reduction (+: rsdnm0, rsdnm1, rsdnm2, rsdnm3, rsdnm4)
#endif
    for (k = 1; k < nz0-1; k++) {
      for (j = jst; j <= jend; j++) {
        for (i = ist; i <= iend; i++) {
          rsdnm0 = rsdnm0 + rsd[0][k][j][i] * rsd[0][k][j][i];
          rsdnm1 = rsdnm1 + rsd[1][k][j][i] * rsd[1][k][j][i];
          rsdnm2 = rsdnm2 + rsd[2][k][j][i] * rsd[2][k][j][i];
          rsdnm3 = rsdnm3 + rsd[3][k][j][i] * rsd[3][k][j][i];
          rsdnm4 = rsdnm4 + rsd[4][k][j][i] * rsd[4][k][j][i];
        }
      }
    }

    rsdnm[0] = rsdnm0;
    rsdnm[1] = rsdnm1;
    rsdnm[2] = rsdnm2;
    rsdnm[3] = rsdnm3;
    rsdnm[4] = rsdnm4;

#ifndef CRPL_COMP
    #pragma omp parallel for private(m)
#elif CRPL_COMP == 0
    #pragma omp parallel for private(m)
#endif
    for (m = 0; m < 5; m++) {
      rsdnm[m] = sqrt ( rsdnm[m] / ( (nx0-2)*(ny0-2)*(nz0-2) ) );
    }
  } // End of omp target

}


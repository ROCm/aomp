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

//---------------------------------------------------------------------
// 
// compute the regular-sparse, block lower triangular solution:
// 
// v <-- ( L-inv ) * v
// 
//---------------------------------------------------------------------
//---------------------------------------------------------------------
// To improve cache performance, second two dimensions padded by 1 
// for even number sizes only.  Only needed in v.
//---------------------------------------------------------------------
void blts (int ldmx, int ldmy, int ldmz, int nx, int ny, int nz, int l,
    double omega)
{
  //---------------------------------------------------------------------
  // local variables
  //---------------------------------------------------------------------
  int i, j, k, m, n;
  double tmp, tmp1;
  int npl = np[l];

#pragma omp target teams map(alloc: a, b, c, d, indxp, jndxp, tv, rsd, tmat)\
        num_teams((npl+127)/128)
  {
    

#ifndef CRPL_COMP
#elif CRPL_COMP == 0
    #pragma omp distribute parallel for private (n, j, i, k)
#endif    
for (n = 1; n <= npl; n++) {
      j = jndxp[l][n];
      i = indxp[l][n];
      k = l - i - j;
      for (m = 0; m < 5; m++) {
        rsd[m][k][j][i] =  rsd[m][k][j][i]
                                        - omega * (  a[0][m][n] * rsd[0][k-1][j][i]
                                                                                 + a[1][m][n] * rsd[1][k-1][j][i]
                                                                                                               + a[2][m][n] * rsd[2][k-1][j][i]
                                                                                                                                             + a[3][m][n] * rsd[3][k-1][j][i]
                                                                                                                                                                           + a[4][m][n] * rsd[4][k-1][j][i] );
      }
    }

#ifndef CRPL_COMP
#elif CRPL_COMP == 0
    #pragma omp distribute parallel for private (n, i, j, k, tmp1, tmp)
#endif
    for (n = 1; n <= npl; n++) {
      j = jndxp[l][n];
      i = indxp[l][n];
      k = l - i - j;
      for (m = 0; m < 5; m++) {
        tv[m][n] =  rsd[m][k][j][i]
                                 - omega * ( b[0][m][n] * rsd[0][k][j-1][i]
                                                                         + c[0][m][n] * rsd[0][k][j][i-1]
                                                                                                     + b[1][m][n] * rsd[1][k][j-1][i]
                                                                                                                                   + c[1][m][n] * rsd[1][k][j][i-1]
                                                                                                                                                               + b[2][m][n] * rsd[2][k][j-1][i]
                                                                                                                                                                                             + c[2][m][n] * rsd[2][k][j][i-1]
                                                                                                                                                                                                                         + b[3][m][n] * rsd[3][k][j-1][i]
                                                                                                                                                                                                                                                       + c[3][m][n] * rsd[3][k][j][i-1]
                                                                                                                                                                                                                                                                                   + b[4][m][n] * rsd[4][k][j-1][i]
                                                                                                                                                                                                                                                                                                                 + c[4][m][n] * rsd[4][k][j][i-1] );
      }

      //---------------------------------------------------------------------
      // diagonal block inversion
      // 
      // forward elimination
      //---------------------------------------------------------------------
      for (m = 0; m < 5; m++) {
        tmat[m][0][n] = d[0][m][n];
        tmat[m][1][n] = d[1][m][n];
        tmat[m][2][n] = d[2][m][n];
        tmat[m][3][n] = d[3][m][n];
        tmat[m][4][n] = d[4][m][n];
      }

      tmp1 = 1.0 / tmat[0][0][n];
      tmp = tmp1 * tmat[1][0][n];
      tmat[1][1][n] =  tmat[1][1][n] - tmp * tmat[0][1][n];
      tmat[1][2][n] =  tmat[1][2][n] - tmp * tmat[0][2][n];
      tmat[1][3][n] =  tmat[1][3][n] - tmp * tmat[0][3][n];
      tmat[1][4][n] =  tmat[1][4][n] - tmp * tmat[0][4][n];
      tv[1][n] = tv[1][n] - tv[0][n] * tmp;

      tmp = tmp1 * tmat[2][0][n];
      tmat[2][1][n] =  tmat[2][1][n] - tmp * tmat[0][1][n];
      tmat[2][2][n] =  tmat[2][2][n] - tmp * tmat[0][2][n];
      tmat[2][3][n] =  tmat[2][3][n] - tmp * tmat[0][3][n];
      tmat[2][4][n] =  tmat[2][4][n] - tmp * tmat[0][4][n];
      tv[2][n] = tv[2][n] - tv[0][n] * tmp;

      tmp = tmp1 * tmat[3][0][n];
      tmat[3][1][n] =  tmat[3][1][n] - tmp * tmat[0][1][n];
      tmat[3][2][n] =  tmat[3][2][n] - tmp * tmat[0][2][n];
      tmat[3][3][n] =  tmat[3][3][n] - tmp * tmat[0][3][n];
      tmat[3][4][n] =  tmat[3][4][n] - tmp * tmat[0][4][n];
      tv[3][n] = tv[3][n] - tv[0][n] * tmp;

      tmp = tmp1 * tmat[4][0][n];
      tmat[4][1][n] =  tmat[4][1][n] - tmp * tmat[0][1][n];
      tmat[4][2][n] =  tmat[4][2][n] - tmp * tmat[0][2][n];
      tmat[4][3][n] =  tmat[4][3][n] - tmp * tmat[0][3][n];
      tmat[4][4][n] =  tmat[4][4][n] - tmp * tmat[0][4][n];
      tv[4][n] = tv[4][n] - tv[0][n] * tmp;

      tmp1 = 1.0 / tmat[1][1][n];
      tmp = tmp1 * tmat[2][1][n];
      tmat[2][2][n] =  tmat[2][2][n] - tmp * tmat[1][2][n];
      tmat[2][3][n] =  tmat[2][3][n] - tmp * tmat[1][3][n];
      tmat[2][4][n] =  tmat[2][4][n] - tmp * tmat[1][4][n];
      tv[2][n] = tv[2][n] - tv[1][n] * tmp;

      tmp = tmp1 * tmat[3][1][n];
      tmat[3][2][n] =  tmat[3][2][n] - tmp * tmat[1][2][n];
      tmat[3][3][n] =  tmat[3][3][n] - tmp * tmat[1][3][n];
      tmat[3][4][n] =  tmat[3][4][n] - tmp * tmat[1][4][n];
      tv[3][n] = tv[3][n] - tv[1][n] * tmp;

      tmp = tmp1 * tmat[4][1][n];
      tmat[4][2][n] =  tmat[4][2][n] - tmp * tmat[1][2][n];
      tmat[4][3][n] =  tmat[4][3][n] - tmp * tmat[1][3][n];
      tmat[4][4][n] =  tmat[4][4][n] - tmp * tmat[1][4][n];
      tv[4][n] = tv[4][n] - tv[1][n] * tmp;

      tmp1 = 1.0 / tmat[2][2][n];
      tmp = tmp1 * tmat[3][2][n];
      tmat[3][3][n] =  tmat[3][3][n] - tmp * tmat[2][3][n];
      tmat[3][4][n] =  tmat[3][4][n] - tmp * tmat[2][4][n];
      tv[3][n] = tv[3][n] - tv[2][n] * tmp;

      tmp = tmp1 * tmat[4][2][n];
      tmat[4][3][n] =  tmat[4][3][n] - tmp * tmat[2][3][n];
      tmat[4][4][n] =  tmat[4][4][n] - tmp * tmat[2][4][n];
      tv[4][n] = tv[4][n] - tv[2][n] * tmp;

      tmp1 = 1.0 / tmat[3][3][n];
      tmp = tmp1 * tmat[4][3][n];
      tmat[4][4][n] =  tmat[4][4][n] - tmp * tmat[3][4][n];
      tv[4][n] = tv[4][n] - tv[3][n] * tmp;

      //---------------------------------------------------------------------
      // back substitution
      //---------------------------------------------------------------------
      rsd[4][k][j][i] = tv[4][n] / tmat[4][4][n];

      tv[3][n] = tv[3][n] 
                       - tmat[3][4][n] * rsd[4][k][j][i];
      rsd[3][k][j][i] = tv[3][n] / tmat[3][3][n];

      tv[2][n] = tv[2][n]
                       - tmat[2][3][n] * rsd[3][k][j][i]
                                                      - tmat[2][4][n] * rsd[4][k][j][i];
      rsd[2][k][j][i] = tv[2][n] / tmat[2][2][n];

      tv[1][n] = tv[1][n]
                       - tmat[1][2][n] * rsd[2][k][j][i]
                                                      - tmat[1][3][n] * rsd[3][k][j][i]
                                                                                     - tmat[1][4][n] * rsd[4][k][j][i];
      rsd[1][k][j][i] = tv[1][n] / tmat[1][1][n];

      tv[0][n] = tv[0][n]
                       - tmat[0][1][n] * rsd[1][k][j][i]
                                                      - tmat[0][2][n] * rsd[2][k][j][i]
                                                                                     - tmat[0][3][n] * rsd[3][k][j][i]
                                                                                                                    - tmat[0][4][n] * rsd[4][k][j][i];
      rsd[0][k][j][i] = tv[0][n] / tmat[0][0][n];
    }

  } // end of omp target data
}


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
// compute the lower triangular part of the jacobian matrix
//---------------------------------------------------------------------
void jacld(int l)
{
  //---------------------------------------------------------------------
  // local variables
  //---------------------------------------------------------------------
  int i, j, k, n;
  double r43;
  double c1345;
  double c34;
  double tmp1, tmp2, tmp3;
  int npl = np[l];

  r43 = ( 4.0 / 3.0 );
  c1345 = C1 * C3 * C4 * C5;
  c34 = C3 * C4;

#pragma omp target teams map (alloc: a, b, c, d, u, indxp, jndxp, rho_i, qs) \
        num_teams((npl+127)/128)
  {

#ifndef CRPL_COMP
#elif CRPL_COMP == 0
    #pragma omp distribute parallel for private( tmp1, tmp2, tmp3, i, j, n, k)
#endif
    for (n = 1; n <= npl; n++) {
      j = jndxp[l][n];
      i = indxp[l][n];
      k = l - i - j;
      //---------------------------------------------------------------------
      // form the block daigonal
      //---------------------------------------------------------------------
      tmp1 = rho_i[k][j][i];
      tmp2 = tmp1 * tmp1;
      tmp3 = tmp1 * tmp2;

      d[0][0][n] =  1.0 + dt * 2.0 * ( tx1 * dx1 + ty1 * dy1 + tz1 * dz1 );
      d[1][0][n] =  0.0;
      d[2][0][n] =  0.0;
      d[3][0][n] =  0.0;
      d[4][0][n] =  0.0;

      d[0][1][n] = -dt * 2.0
          * ( tx1 * r43 + ty1 + tz1 ) * c34 * tmp2 * u[1][k][j][i];
      d[1][1][n] =  1.0
          + dt * 2.0 * c34 * tmp1 * ( tx1 * r43 + ty1 + tz1 )
          + dt * 2.0 * ( tx1 * dx2 + ty1 * dy2 + tz1 * dz2 );
      d[2][1][n] = 0.0;
      d[3][1][n] = 0.0;
      d[4][1][n] = 0.0;

      d[0][2][n] = -dt * 2.0 
          * ( tx1 + ty1 * r43 + tz1 ) * c34 * tmp2 * u[2][k][j][i];
      d[1][2][n] = 0.0;
      d[2][2][n] = 1.0
          + dt * 2.0 * c34 * tmp1 * ( tx1 + ty1 * r43 + tz1 )
          + dt * 2.0 * ( tx1 * dx3 + ty1 * dy3 + tz1 * dz3 );
      d[3][2][n] = 0.0;
      d[4][2][n] = 0.0;

      d[0][3][n] = -dt * 2.0
          * ( tx1 + ty1 + tz1 * r43 ) * c34 * tmp2 * u[3][k][j][i];
      d[1][3][n] = 0.0;
      d[2][3][n] = 0.0;
      d[3][3][n] = 1.0
          + dt * 2.0 * c34 * tmp1 * ( tx1 + ty1 + tz1 * r43 )
          + dt * 2.0 * ( tx1 * dx4 + ty1 * dy4 + tz1 * dz4 );
      d[4][3][n] = 0.0;

      d[0][4][n] = -dt * 2.0
          * ( ( ( tx1 * ( r43*c34 - c1345 )
              + ty1 * ( c34 - c1345 )
              + tz1 * ( c34 - c1345 ) ) * ( u[1][k][j][i]*u[1][k][j][i] )
              + ( tx1 * ( c34 - c1345 )
                  + ty1 * ( r43*c34 - c1345 )
                  + tz1 * ( c34 - c1345 ) ) * ( u[2][k][j][i]*u[2][k][j][i] )
                  + ( tx1 * ( c34 - c1345 )
                      + ty1 * ( c34 - c1345 )
                      + tz1 * ( r43*c34 - c1345 ) ) * (u[3][k][j][i]*u[3][k][j][i])
          ) * tmp3
              + ( tx1 + ty1 + tz1 ) * c1345 * tmp2 * u[4][k][j][i] );

      d[1][4][n] = dt * 2.0 * tmp2 * u[1][k][j][i]
                                                * ( tx1 * ( r43*c34 - c1345 )
                                                    + ty1 * (     c34 - c1345 )
                                                    + tz1 * (     c34 - c1345 ) );
      d[2][4][n] = dt * 2.0 * tmp2 * u[2][k][j][i]
                                                * ( tx1 * ( c34 - c1345 )
                                                    + ty1 * ( r43*c34 -c1345 )
                                                    + tz1 * ( c34 - c1345 ) );
      d[3][4][n] = dt * 2.0 * tmp2 * u[3][k][j][i]
                                                * ( tx1 * ( c34 - c1345 )
                                                    + ty1 * ( c34 - c1345 )
                                                    + tz1 * ( r43*c34 - c1345 ) );
      d[4][4][n] = 1.0
          + dt * 2.0 * ( tx1  + ty1 + tz1 ) * c1345 * tmp1
          + dt * 2.0 * ( tx1 * dx5 +  ty1 * dy5 +  tz1 * dz5 );

      //---------------------------------------------------------------------
      // form the first block sub-diagonal
      //---------------------------------------------------------------------
      tmp1 = rho_i[k-1][j][i];
      tmp2 = tmp1 * tmp1;
      tmp3 = tmp1 * tmp2;

      a[0][0][n] = - dt * tz1 * dz1;
      a[1][0][n] =   0.0;
      a[2][0][n] =   0.0;
      a[3][0][n] = - dt * tz2;
      a[4][0][n] =   0.0;

      a[0][1][n] = - dt * tz2
          * ( - ( u[1][k-1][j][i]*u[3][k-1][j][i] ) * tmp2 )
          - dt * tz1 * ( - c34 * tmp2 * u[1][k-1][j][i] );
      a[1][1][n] = - dt * tz2 * ( u[3][k-1][j][i] * tmp1 )
            - dt * tz1 * c34 * tmp1
            - dt * tz1 * dz2;
      a[2][1][n] = 0.0;
      a[3][1][n] = - dt * tz2 * ( u[1][k-1][j][i] * tmp1 );
      a[4][1][n] = 0.0;

      a[0][2][n] = - dt * tz2
          * ( - ( u[2][k-1][j][i]*u[3][k-1][j][i] ) * tmp2 )
          - dt * tz1 * ( - c34 * tmp2 * u[2][k-1][j][i] );
      a[1][2][n] = 0.0;
      a[2][2][n] = - dt * tz2 * ( u[3][k-1][j][i] * tmp1 )
            - dt * tz1 * ( c34 * tmp1 )
            - dt * tz1 * dz3;
      a[3][2][n] = - dt * tz2 * ( u[2][k-1][j][i] * tmp1 );
      a[4][2][n] = 0.0;

      a[0][3][n] = - dt * tz2
          * ( - ( u[3][k-1][j][i] * tmp1 ) * ( u[3][k-1][j][i] * tmp1 )
              + C2 * qs[k-1][j][i] * tmp1 )
              - dt * tz1 * ( - r43 * c34 * tmp2 * u[3][k-1][j][i] );
      a[1][3][n] = - dt * tz2
          * ( - C2 * ( u[1][k-1][j][i] * tmp1 ) );
      a[2][3][n] = - dt * tz2
          * ( - C2 * ( u[2][k-1][j][i] * tmp1 ) );
      a[3][3][n] = - dt * tz2 * ( 2.0 - C2 )
            * ( u[3][k-1][j][i] * tmp1 )
            - dt * tz1 * ( r43 * c34 * tmp1 )
            - dt * tz1 * dz4;
      a[4][3][n] = - dt * tz2 * C2;

      a[0][4][n] = - dt * tz2
          * ( ( C2 * 2.0 * qs[k-1][j][i] - C1 * u[4][k-1][j][i] )
              * u[3][k-1][j][i] * tmp2 )
              - dt * tz1
              * ( - ( c34 - c1345 ) * tmp3 * (u[1][k-1][j][i]*u[1][k-1][j][i])
                  - ( c34 - c1345 ) * tmp3 * (u[2][k-1][j][i]*u[2][k-1][j][i])
                  - ( r43*c34 - c1345 )* tmp3 * (u[3][k-1][j][i]*u[3][k-1][j][i])
                  - c1345 * tmp2 * u[4][k-1][j][i] );
      a[1][4][n] = - dt * tz2
          * ( - C2 * ( u[1][k-1][j][i]*u[3][k-1][j][i] ) * tmp2 )
          - dt * tz1 * ( c34 - c1345 ) * tmp2 * u[1][k-1][j][i];
      a[2][4][n] = - dt * tz2
          * ( - C2 * ( u[2][k-1][j][i]*u[3][k-1][j][i] ) * tmp2 )
          - dt * tz1 * ( c34 - c1345 ) * tmp2 * u[2][k-1][j][i];
      a[3][4][n] = - dt * tz2
          * ( C1 * ( u[4][k-1][j][i] * tmp1 )
              - C2 * ( qs[k-1][j][i] * tmp1
                  + u[3][k-1][j][i]*u[3][k-1][j][i] * tmp2 ) )
                  - dt * tz1 * ( r43*c34 - c1345 ) * tmp2 * u[3][k-1][j][i];
      a[4][4][n] = - dt * tz2
          * ( C1 * ( u[3][k-1][j][i] * tmp1 ) )
          - dt * tz1 * c1345 * tmp1
          - dt * tz1 * dz5;

      //---------------------------------------------------------------------
      // form the second block sub-diagonal
      //---------------------------------------------------------------------
      tmp1 = rho_i[k][j-1][i];
      tmp2 = tmp1 * tmp1;
      tmp3 = tmp1 * tmp2;

      b[0][0][n] = - dt * ty1 * dy1;
      b[1][0][n] =   0.0;
      b[2][0][n] = - dt * ty2;
      b[3][0][n] =   0.0;
      b[4][0][n] =   0.0;

      b[0][1][n] = - dt * ty2
          * ( - ( u[1][k][j-1][i]*u[2][k][j-1][i] ) * tmp2 )
          - dt * ty1 * ( - c34 * tmp2 * u[1][k][j-1][i] );
      b[1][1][n] = - dt * ty2 * ( u[2][k][j-1][i] * tmp1 )
            - dt * ty1 * ( c34 * tmp1 )
            - dt * ty1 * dy2;
      b[2][1][n] = - dt * ty2 * ( u[1][k][j-1][i] * tmp1 );
      b[3][1][n] = 0.0;
      b[4][1][n] = 0.0;

      b[0][2][n] = - dt * ty2
          * ( - ( u[2][k][j-1][i] * tmp1 ) * ( u[2][k][j-1][i] * tmp1 )
              + C2 * ( qs[k][j-1][i] * tmp1 ) )
              - dt * ty1 * ( - r43 * c34 * tmp2 * u[2][k][j-1][i] );
      b[1][2][n] = - dt * ty2
          * ( - C2 * ( u[1][k][j-1][i] * tmp1 ) );
      b[2][2][n] = - dt * ty2 * ( (2.0 - C2) * (u[2][k][j-1][i] * tmp1) )
            - dt * ty1 * ( r43 * c34 * tmp1 )
            - dt * ty1 * dy3;
      b[3][2][n] = - dt * ty2 * ( - C2 * ( u[3][k][j-1][i] * tmp1 ) );
      b[4][2][n] = - dt * ty2 * C2;

      b[0][3][n] = - dt * ty2
          * ( - ( u[2][k][j-1][i]*u[3][k][j-1][i] ) * tmp2 )
          - dt * ty1 * ( - c34 * tmp2 * u[3][k][j-1][i] );
      b[1][3][n] = 0.0;
      b[2][3][n] = - dt * ty2 * ( u[3][k][j-1][i] * tmp1 );
      b[3][3][n] = - dt * ty2 * ( u[2][k][j-1][i] * tmp1 )
            - dt * ty1 * ( c34 * tmp1 )
            - dt * ty1 * dy4;
      b[4][3][n] = 0.0;

      b[0][4][n] = - dt * ty2
          * ( ( C2 * 2.0 * qs[k][j-1][i] - C1 * u[4][k][j-1][i] )
              * ( u[2][k][j-1][i] * tmp2 ) )
              - dt * ty1
              * ( - (     c34 - c1345 )*tmp3*(u[1][k][j-1][i]*u[1][k][j-1][i])
                  - ( r43*c34 - c1345 )*tmp3*(u[2][k][j-1][i]*u[2][k][j-1][i])
                  - (     c34 - c1345 )*tmp3*(u[3][k][j-1][i]*u[3][k][j-1][i])
                  - c1345*tmp2*u[4][k][j-1][i] );
      b[1][4][n] = - dt * ty2
          * ( - C2 * ( u[1][k][j-1][i]*u[2][k][j-1][i] ) * tmp2 )
          - dt * ty1 * ( c34 - c1345 ) * tmp2 * u[1][k][j-1][i];
      b[2][4][n] = - dt * ty2
          * ( C1 * ( u[4][k][j-1][i] * tmp1 )
              - C2 * ( qs[k][j-1][i] * tmp1
                  + u[2][k][j-1][i]*u[2][k][j-1][i] * tmp2 ) )
                  - dt * ty1 * ( r43*c34 - c1345 ) * tmp2 * u[2][k][j-1][i];
      b[3][4][n] = - dt * ty2
          * ( - C2 * ( u[2][k][j-1][i]*u[3][k][j-1][i] ) * tmp2 )
          - dt * ty1 * ( c34 - c1345 ) * tmp2 * u[3][k][j-1][i];
      b[4][4][n] = - dt * ty2
          * ( C1 * ( u[2][k][j-1][i] * tmp1 ) )
          - dt * ty1 * c1345 * tmp1
          - dt * ty1 * dy5;

      //---------------------------------------------------------------------
      // form the third block sub-diagonal
      //---------------------------------------------------------------------
      tmp1 = rho_i[k][j][i-1];
      tmp2 = tmp1 * tmp1;
      tmp3 = tmp1 * tmp2;

      c[0][0][n] = - dt * tx1 * dx1;
      c[1][0][n] = - dt * tx2;
      c[2][0][n] =   0.0;
      c[3][0][n] =   0.0;
      c[4][0][n] =   0.0;

      c[0][1][n] = - dt * tx2
          * ( - ( u[1][k][j][i-1] * tmp1 ) * ( u[1][k][j][i-1] * tmp1 )
              + C2 * qs[k][j][i-1] * tmp1 )
              - dt * tx1 * ( - r43 * c34 * tmp2 * u[1][k][j][i-1] );
      c[1][1][n] = - dt * tx2
          * ( ( 2.0 - C2 ) * ( u[1][k][j][i-1] * tmp1 ) )
          - dt * tx1 * ( r43 * c34 * tmp1 )
          - dt * tx1 * dx2;
      c[2][1][n] = - dt * tx2
          * ( - C2 * ( u[2][k][j][i-1] * tmp1 ) );
      c[3][1][n] = - dt * tx2
          * ( - C2 * ( u[3][k][j][i-1] * tmp1 ) );
      c[4][1][n] = - dt * tx2 * C2;

      c[0][2][n] = - dt * tx2
          * ( - ( u[1][k][j][i-1] * u[2][k][j][i-1] ) * tmp2 )
          - dt * tx1 * ( - c34 * tmp2 * u[2][k][j][i-1] );
      c[1][2][n] = - dt * tx2 * ( u[2][k][j][i-1] * tmp1 );
      c[2][2][n] = - dt * tx2 * ( u[1][k][j][i-1] * tmp1 )
            - dt * tx1 * ( c34 * tmp1 )
            - dt * tx1 * dx3;
      c[3][2][n] = 0.0;
      c[4][2][n] = 0.0;

      c[0][3][n] = - dt * tx2
          * ( - ( u[1][k][j][i-1]*u[3][k][j][i-1] ) * tmp2 )
          - dt * tx1 * ( - c34 * tmp2 * u[3][k][j][i-1] );
      c[1][3][n] = - dt * tx2 * ( u[3][k][j][i-1] * tmp1 );
      c[2][3][n] = 0.0;
      c[3][3][n] = - dt * tx2 * ( u[1][k][j][i-1] * tmp1 )
            - dt * tx1 * ( c34 * tmp1 ) - dt * tx1 * dx4;
      c[4][3][n] = 0.0;

      c[0][4][n] = - dt * tx2
          * ( ( C2 * 2.0 * qs[k][j][i-1] - C1 * u[4][k][j][i-1] )
              * u[1][k][j][i-1] * tmp2 )
              - dt * tx1
              * ( - ( r43*c34 - c1345 ) * tmp3 * ( u[1][k][j][i-1]*u[1][k][j][i-1] )
                  - (     c34 - c1345 ) * tmp3 * ( u[2][k][j][i-1]*u[2][k][j][i-1] )
                  - (     c34 - c1345 ) * tmp3 * ( u[3][k][j][i-1]*u[3][k][j][i-1] )
                  - c1345 * tmp2 * u[4][k][j][i-1] );
      c[1][4][n] = - dt * tx2
          * ( C1 * ( u[4][k][j][i-1] * tmp1 )
              - C2 * ( u[1][k][j][i-1]*u[1][k][j][i-1] * tmp2
                  + qs[k][j][i-1] * tmp1 ) )
                  - dt * tx1 * ( r43*c34 - c1345 ) * tmp2 * u[1][k][j][i-1];
      c[2][4][n] = - dt * tx2
          * ( - C2 * ( u[2][k][j][i-1]*u[1][k][j][i-1] ) * tmp2 )
          - dt * tx1 * (  c34 - c1345 ) * tmp2 * u[2][k][j][i-1];
      c[3][4][n] = - dt * tx2
          * ( - C2 * ( u[3][k][j][i-1]*u[1][k][j][i-1] ) * tmp2 )
          - dt * tx1 * (  c34 - c1345 ) * tmp2 * u[3][k][j][i-1];
      c[4][4][n] = - dt * tx2
          * ( C1 * ( u[1][k][j][i-1] * tmp1 ) )
          - dt * tx1 * c1345 * tmp1
          - dt * tx1 * dx5;
    }
            
} // end of omp target data
        
}


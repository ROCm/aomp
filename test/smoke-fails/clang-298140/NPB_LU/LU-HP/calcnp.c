
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

void calcnp(int lst, int lend)
{
	int i, j, k, l;

  //#pragma acc kernels present(indxp, jndxp, np)
  {
    //#pragma acc loop private(i,j,k)
	for(l = lst; l <= lend; l++)
	{
		int n = 0;
		for(j = max(1, l-nx-nz+2); j <= min(ny-2, l-2); j++)
		{
			for(i = max(1, l-j-nz+2); i <= min(nx-2, l-j-1); i++)
			{
				k = l - i - j;
				if(k >= 1 && k < nz - 1)
				{
					n = n + 1;
					indxp[l][n] = i;
					jndxp[l][n] = j;
				}
			}
		}
		np[l] = n;
	}
  }
}

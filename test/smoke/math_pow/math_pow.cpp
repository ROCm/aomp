#include <cmath>
#include <omp.h>
#include <stdio.h>
#define RES_SIZE 14

int main(int argc, char **argv)
{
  int ibase   = 2 ;
  int iexp    = 3 ;
  double base = 2.0 ;
  double exp  = 3.0 ;
  float fbase = 2.0 ;
  float fexp  = 3.0 ;
  unsigned long ulval = 2;
  double res[RES_SIZE];
  float  fres[RES_SIZE];
  int ires[RES_SIZE];
  unsigned long ulres[RES_SIZE];
  #pragma omp target map(from:res[0:RES_SIZE],fres[0:RES_SIZE],ires[0:RES_SIZE],ulres[0:RES_SIZE])
  {
	  // Double results 
    res[0] = pow(base, exp);
    res[1] = pow(fbase, exp);
    res[2] = pow(base, iexp);
    res[3] = pow(base, fexp);
    res[4] = pow(ibase, iexp);
    res[5] = pow(ibase, exp);

    res[6] = std::pow(base, exp);
    res[7] = std::pow(fbase, exp);
    res[8] = std::pow(base, iexp);
    res[9] = std::pow(base, fexp);
    res[10] = std::pow(ibase, iexp);
    res[11] = std::pow(ibase, exp);

	  // float results 
    fres[0] = pow(fbase, fexp);
    fres[1] = pow(fbase, iexp);
    fres[2] = pow(ibase, iexp);
    fres[3] = pow(ibase, fexp);
    fres[4] = std::pow(fbase, fexp);
    fres[5] = std::pow(fbase, iexp);
    fres[6] = std::pow(ibase, iexp);
    fres[7] = std::pow(ibase, fexp);

	  // integer results 
    ires[0] = pow(ibase, iexp);
    ires[1] = std::pow(ibase, iexp);

	  // unsigned long results
    res[12] = sqrt(ulval);
  }

  printf(" Double = Double**Double    result = %f\n", res[0]);
  printf(" Double = Float**Double     result = %f\n", res[1]);
  printf(" Double = Double**Integer   result = %f\n", res[2]);
  printf(" Double = Double**Float     result = %f\n", res[3]);
  printf(" Double = Integer**Integer  result = %f\n", res[4]);
  printf(" Double = Integer**double   result = %f\n", res[5]);
  printf(" With std::\n");
  printf(" Double = Double**Double    result = %f\n", res[6]);
  printf(" Double = Float**Double     result = %f\n", res[7]);
  printf(" Double = Double**Integer   result = %f\n", res[8]);
  printf(" Double = Double**Float     result = %f\n", res[9]);
  printf(" Double = Integer**Integer  result = %f\n", res[10]);
  printf(" Double = Integer**double   result = %f\n", res[11]);

  printf(" Floats\n");
  printf(" Float = Float**Float       result = %f\n",fres[0]);
  printf(" Float = Float**Integer     result = %f\n",fres[1]);
  printf(" Float = Integer**Integer   result = %f\n",fres[2]);
  printf(" Float = Integer**Float     result = %f\n",fres[3]);
  printf(" Floats With std::\n");
  printf(" Float = Float**Float       result = %f\n",fres[4]);
  printf(" Float = Float**Integer     result = %f\n",fres[5]);
  printf(" Float = Integer**Integer   result = %f\n",fres[6]);
  printf(" Float = Integer**Float     result = %f\n",fres[7]);

  printf(" Integer = Integer**Integer result = %d\n",ires[0]);
  printf(" With std::\n");
  printf(" Integer = Integer**Integer result = %d\n",ires[1]);
  printf(" Double = sqrt(ulong 2)     result = %f\n", res[12]);

  return 0;
}

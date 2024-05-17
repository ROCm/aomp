#include <omp.h>
#include <iostream>
#include <cmath>
using namespace std;

double E[4] = { 125.0, 25.0, 1.0, 5.0};

volatile double val=5.0;
int main()
{
       int i;
       int res = 0;
       #pragma omp parallel for
       for (i=0; i < 4; i++)
       {
         double powval;
         powval=pow(val,i);
         printf ("pow(%lf, %i) = %lf\n", val, i, powval);
	 if (E[i] != powval) res = 1;
       }
 
    return (res);
}

#include <omp.h>
#include <iostream>
#include <cmath>
using namespace std;

volatile double val=5.0;
int main()
{
       int i;

       #pragma omp parallel for
       for (i=0; i < 4; i++)
       {
         double powval;
         powval=pow(val,i);
         printf ("pow(%lf, %i) = %lf\n", val, i, powval);
       }

    return (0);
}

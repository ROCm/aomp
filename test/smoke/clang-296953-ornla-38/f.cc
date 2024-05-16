#include "global.h"

#pragma omp declare target
double f(const double x)
{
    return aaa*x;
}
#pragma omp end declare target

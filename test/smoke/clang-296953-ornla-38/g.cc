#include "global.h"

#pragma omp declare target
double g(const double x)
{
    return aaa*x*x;
}
#pragma omp end declare target

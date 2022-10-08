#include "fg.h"

#include <iostream>
#include <omp.h>

int main(int argc, char* argv[])
{
    const int N = 10;

    double* result = new double[N];
#pragma omp target map(tofrom : result[:N])
{
#pragma omp teams distribute parallel for
    for (int i = 0; i < N; i++)
    {
        if( omp_is_initial_device() )continue;
        result[i] = f(i) + g(i);
    }
}

    for(int i = 0; i < N; i++)
    {
        std::cout<<"result = "<<result[i]<<std::endl;
    }

    delete[] result;
}


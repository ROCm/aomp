#include <stdio.h>

#include <cstddef>

#include <type_traits>

template<typename Type>

void set_to_zero(Type * data, int N){

        //#pragma omp target teams distribute parallel for if(target: std::is_arithmetic_v<Type> && N > 200)

        _Pragma("omp target teams distribute parallel for if(target: std::is_arithmetic_v<Type> && N > 200)")

        for (int i = 0; i < N; ++i) data[i] = 0.0001;

}

int main(){

    int N = 1024*100;

    double *B1 = new double[N];

    set_to_zero<double>(B1,N);

    double sum = 0.0;

    #pragma omp target teams distribute parallel for map(tofrom:sum)  if(N > 1000)  reduction(+:sum)

    for (int i = 0; i < N; ++i)

            sum+=B1[i];

    printf("sum = %g, expected =%g\n",sum,0.0001*N);

    delete[] B1;

    return 0;

}


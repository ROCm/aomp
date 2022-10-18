#include <math.h>
#include <stdlib.h>
#include <omp.h>


int main(int argc, char const * argv[]) {
    const int N = 1000;
    int array[N];
    for (int i=0; i<N; i++){
        array[i] = i;
    }
    #pragma omp target
    {
        #pragma omp parallel for
        for (int i=-N; i<N; i++){
            array[i] += array[(int)abs(i)];
        }
    }
    return 0;
}

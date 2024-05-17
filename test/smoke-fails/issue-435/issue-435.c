#include <math.h>
#include <stdlib.h>
#include <omp.h>
#include <stdio.h>


int simplesum(int argc, char const * argv[]) {
    const int N = 1000;
    int array[N];
    for (int i=0; i<N; i++){
        array[i] = i;
    }

    #pragma omp target map(tofrom: array[:N])
    {
        #pragma omp parallel for
        for (int i=0; i<N; i++) {
            array[(int)abs(-i)] += (int)abs(-i);
        }
    }

    int err = 0;
    for (int i=0; i<N; i++){
        if (array[i] != 2*i) {
            err += 1;
        }
    }
    if (!err) {
        printf("Success\n");
        return 0;
    }

    printf("Fail\n");
    return 1;
}

int main(int argc, char const * argv[]) {
  return simplesum(argc, argv);
}

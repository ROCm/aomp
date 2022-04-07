// compile with:
//    clang++ -g -O0 -fopenmp --offload-arch=gfx908 -o reproducer reproducer.cc
//
// expected output:
//    $ ./reproducer
//    line 34: [host]   x: 0x273f60
//    line 43: [device] x: 0x15554a400000
//    line 49: [host]   x: 0x15554a400000
//    line 55: [host]   x: 0x15554a400000
//
// actual output
//    $ ./reproducer
//    line 33: [host]   x: 0x273f60
//    line 42: [device] x: 0x15554a400000
//    line 48: [host]   x: 0x15554a400000
//    [1]    3915739 segmentation fault (core dumped)  ./reproducer

#include <stdio.h>
#include <stdlib.h>

#include <omp.h>

int main(int argc, char * argv[]) {
    const size_t count = 256;

    float * x;

    x = (float *) malloc(sizeof(float) * count);

    // make x available on the GPU
    #pragma omp target data map(to:x[0:count])
    {
        fprintf(stderr, "line %d: [host]   x: %p\n", __LINE__, x);

        // determine the real GPU address of x as a uintptr_t
        // get it from the device
        uintptr_t px = 0;
        #pragma omp target map(from:px)
        {
            px = reinterpret_cast<uintptr_t>(x);
        }
        fprintf(stderr, "line %d: [device] x: %p\n", __LINE__,
               reinterpret_cast<void *>(px));

        // use OpenMP pointer translation to get device pointer
        #pragma omp target data use_device_ptr(x)
        {
            fprintf(stderr, "line %d: [host]   x: %p\n", __LINE__, x);
        }

        // use OpenMP pointer translation to get device address
        #pragma omp target data use_device_addr(x)
        {
            fprintf(stderr, "line %d: [host]   x: %p\n", __LINE__, x);
        }
    }
    return EXIT_SUCCESS;
}

// Test case from https://github.com/ROCm-Developer-Tools/aomp/issues/531
// [OMPT] Device tracing interface buffer records have fixed thread_id of 0

#include <omp.h>
#include <stdio.h>
#include "callbacks.h"

// Map of devices traced
DeviceMapPtr_t DeviceMapPtr;

int main( void )
{
#pragma omp parallel num_threads(2)
    {
        int M[10];
#pragma omp target enter data map(to: M[:10]) 
#pragma omp target map(tofrom: M) 
	{
#pragma omp teams distribute parallel for simd
            for(int i = 0; i < 10; ++i)
            {
                M[i] = i;
            }
        }
#pragma omp target exit data map(from: M[:10]) 
    }
    return 0;
}

/// CHECK-NOT: thread_id=0

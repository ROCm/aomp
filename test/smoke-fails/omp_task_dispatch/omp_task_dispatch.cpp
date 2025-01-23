#include <cstdio>
#include <cstdlib>
#include <omp.h>
#define __HIP_PLATFORM_AMD__
#include <hip/hip_runtime.h>


void launch_hip_add(hipStream_t *stream, int *a);

int dep = 0;

void callback_function(hipStream_t stream, hipError_t err, void *event) {
    omp_fulfill_event(*(omp_event_handle_t*)event);
}

// If OpenMP is called first and then hip, the answer is: (1 * 2) + 2 = 4
// If hip is called first and then OpenMP, the answer is: (1 + 2) * 2 = 6

int main() {

    hipStream_t stream;
    hipStreamCreate(&stream);

    int *a = new int[1];
    a[0] = 1;

    omp_event_handle_t event, hip_event;
    hipStreamCallback_t callback = &callback_function;

#pragma omp target enter data map(to:a[0])

#pragma omp task depend(out:dep) detach(event)
{
#pragma omp target data use_device_ptr(a)
    launch_hip_add(&stream, a);
    hipStreamAddCallback(stream, callback, &event, 0);
}

#pragma omp target depend(in:dep)
{
printf("Calling OpenMP kernel\n");
a[0] *= 2.0;
}

    hipDeviceSynchronize();
// cray deadlocks with this statement present, but gives the correct output when disabled.
#pragma omp taskwait

#pragma omp target update from(a[0])

    if (a[0] == 4) printf("OpenMP kernel updated a[0] first. Final value is a[0] = %d.\n", a[0]);
    else if (a[0] == 6) printf("HIP kernel updated a[0] first. Final value is a[0] = %d.\n", a[0]);
    else printf("Not all kernels completed\n");

    hipStreamDestroy(stream);
    hipDeviceSynchronize();
    int rc=-1;
    if (a[0] == 6)
      rc = 0;
    else 
      rc = 1;
#pragma omp target exit data map(delete:a)
    delete[] a;
    return rc;
}

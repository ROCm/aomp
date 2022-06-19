#include <assert.h>
#include <hip_runtime.h>
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>

#define PAGE_SIZE sysconf(_SC_PAGESIZE)

#define GB (1024 * 1024 * 1024)
const size_t SIZE = .5 * 600 * 1024L * 1024L;
const int NUM_THREADS = 1;

#define HIP_RC(hipCall)                                                     \
    {                                                                       \
        hipError_t e = hipCall;                                             \
        if (e != hipSuccess)                                                \
        {                                                                   \
            e = hipGetLastError();                                          \
            fprintf(stdout, "%s:%d -- %s returned %d:%s\n ",                \
                    __FILE__, __LINE__, #hipCall, e, hipGetErrorString(e)); \
            abort();                                                        \
        }                                                                   \
    }

#ifdef USE_USM
#pragma omp requires unified_shared_memory
#endif

void run_test(double *p, const char *tag, bool managed_memory = false, bool associated = false)
{
    double mem_size = (double)SIZE * sizeof(double) / GB;

    for (int j = 0; j < 2; ++j)
    {
        if (not managed_memory)
        {
            double start = omp_get_wtime();
            if (not associated)
            {
#pragma omp target enter data map(always, to \
                                  : p[:SIZE])
            }
            double end = omp_get_wtime();
            printf("%s data MAP (to:) time = %.6lf Sec.  BW = %.6lf GB/S\n",
                   tag,
                   end - start,
                   mem_size / (end - start));
        }
        {
            double start = omp_get_wtime();
#pragma omp parallel for num_threads(NUM_THREADS)
            for (size_t i = 0; i < SIZE; i++)
            {
                p[i] = 1.0;
            }
            double end = omp_get_wtime();
            printf("%s CPU LOOP time before = %.6lf Sec.  BW = %.6lf GB/S\n",
                   tag,
                   end - start,
                   mem_size / (end - start));
        }
        for (int k = 0; k < 2; ++k)
        {
            double start = omp_get_wtime();
            double start_kernel = 0.0;
            double end_kernel = 0.0;
            if (not managed_memory)
            {
#pragma omp target update to(p[:SIZE])
                double end = omp_get_wtime();
                printf("%s (%d) HTOD time = %.6lf Sec.  BW = %.6lf GB/S\n",
                       tag, k,
                       end - start,
                       mem_size / (end - start));
                start_kernel = omp_get_wtime();
#pragma omp target teams distribute parallel for
                for (size_t i = 0; i < SIZE; i++)
                {
                    p[i] = 2.0;
                }
                end_kernel = omp_get_wtime();
            }
            else
            {
                start_kernel = omp_get_wtime();
#pragma omp target teams distribute parallel for is_device_ptr(p)
                for (size_t i = 0; i < SIZE; i++)
                {
                    p[i] = 2.0;
                }
                end_kernel = omp_get_wtime();
            }
            printf("%s (%d) GPU KERNEL time = %.6lf Sec.  BW = %.6lf GB/S\n",
                   tag, k,
                   end_kernel - start_kernel,
                   mem_size / (end_kernel - start_kernel));

            double start1 = omp_get_wtime();
            double end1;
            if (not managed_memory)
            {
#pragma omp target update from(p[:SIZE])
                end1 = omp_get_wtime();
                printf("%s (%d) DTOH time = %.6lf Sec.  BW = %.6lf GB/S\n",
                       tag, k,
                       end1 - start1,
                       mem_size / (end1 - start1));
            }
            else
            {
                end1 = omp_get_wtime();
            }
            printf("%s (%d) GPU TOTAL time = %.6lf Sec.  BW = %.6lf GB/S\n",
                   tag, k,
                   end1 - start,
                   mem_size / (end1 - start));
        }
        if (not managed_memory && not associated)
        {
#pragma omp target exit data map(delete \
                                 : p[:SIZE])
        }

        {
            double start = omp_get_wtime();
#pragma omp parallel for num_threads(NUM_THREADS)
            for (size_t i = 0; i < SIZE; i++)
            {
                p[i] = 1.0;
            }
            double end = omp_get_wtime();
            printf("%s CPU LOOP time after = %.6lf  BW = %.6lf GB/S\n",
                   tag,
                   end - start,
                   mem_size / (end - start));
        }
        printf("\n");
    }
    printf("\n");
}

int main()
{
    int dummy = 0;

    printf("TOtal Memory = %.6lf GB\n\n", (double)SIZE * sizeof(double) / GB);

// Initialize device runtime and state
#pragma omp target
    {
        dummy += 1;
    }

    if (1)
    {
        printf("Running omp_default_mem_alloc() test\n====================\n");
        double *p = NULL;
        p = (double *)omp_alloc(SIZE * sizeof(double), omp_default_mem_alloc);

        if (!p)
            abort();
        run_test(p, "omp_default_mem_alloc");
        printf("\n");
        omp_free(p);
    }

    if (1)
    {
        printf("Running malloc() Mapped test\n====================\n");
        double *p = new (std::align_val_t(sizeof(PAGE_SIZE))) double[SIZE];
        if (!p)
            abort();
        run_test(p, "malloc() Mapped");
        printf("\n");
        free(p);
    }

    return 0;
}

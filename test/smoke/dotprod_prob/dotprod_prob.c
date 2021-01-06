#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <time.h>
#include <omp.h>

#ifndef TYPE
#define TYPE float
#endif

#define MAX_SOURCE_SIZE 10000

#define LO -20
#define HI 20
#define SEED_MULT 11

static _Thread_local int rng_init = 0;
static _Thread_local struct drand48_data rng_buf;

#define print_any(X) _Generic((X),                      \
                              int8_t: print_int8_t,     \
                              int32_t: print_int32_t,   \
                              int64_t: print_int64_t,   \
                              float: print_float)(X)

int print_int8_t(int8_t i)
{
    return printf("%d", i);
}

int print_int32_t(int32_t i)
{
    return printf("%d", i);
}

int print_int64_t(int32_t i)
{
    return printf("%d", i);
}

int print_float(float f)
{
    return printf("%f", f);
}

bool
is_omp_using_gpu() {
    int A[1] = {-1};
#pragma omp target map(tofrom: A[:1])
    {
        A[0] = omp_is_initial_device();
    }
    if (!A[0]) return true;
    return false;
}

void
print_sys_info()
{
    #pragma omp target
    {
        printf("Number of devices %d\n", omp_get_num_devices());
        printf("Default device %d\n", omp_get_default_device());
        printf("Initial device %d\n", omp_get_initial_device());
        printf("Max threads %d\n", omp_get_max_threads());
        printf("OMP able to use GPU? %d\n", is_omp_using_gpu());
        if (omp_is_initial_device()) {
            printf("Hello World from Host.\n");
        } else {
            printf("Hello World from Accelerator(s).\n");
        }
    }
}

void
process_args(int argc, char *argv[], int *n, int *iterations,
             long *seed, bool *verbose)
{
    int opt;
    while ((opt = getopt(argc, argv, "n:r:i:v:s")) != -1) {
        switch (opt) {
        case 'n':
            *n = atoi(optarg);
            break;
        case 'r':
            *seed = atol(optarg);
            break;
        case 'i':
            *iterations = atoi(optarg);
            break;
        case 'v':
            *verbose = (atoi(optarg) > 0) ? true : false;
            break;
        case 's':
            print_sys_info();
            break;
        default: /* '?' */
            fprintf(stderr, "Usage: %s [-n <int>] [-r <int>]"
                    "[-iterations <int>]\n"
                    "where:\n"
                    "\tn: size of vector\n",
                    argv[0]);
            exit(EXIT_FAILURE);
        }
    }
}

void print_vector(const TYPE vector[], int n)
{
    for (int i = 0; i < n; i++) {
        print_any(vector[i]);
        printf("\n");
    }
}

TYPE
dotprod_one_core(const TYPE v1[], const TYPE v2[], int n)
{
    TYPE total = 0;
    for (int i = 0; i < n; i++)
        total += v1[i] * v2[i];
    return total;
}

TYPE
dotprod_multicore(const TYPE v1[], const TYPE v2[], int n)
{
	TYPE total = 0;
        int i;

#pragma omp parallel for reduction(+:total)
        for (i = 0; i < n; i++)
            total += v1[i] * v2[i];

        return total;
}


TYPE
dotprod_target(const TYPE v1[], const TYPE v2[], int n)
{
	TYPE total = 0;
        int i;

#pragma omp target teams distribute parallel for reduction(+:total) map(tofrom: v1[:10], v2[:10])
	for (i = 0; i < n; i++)
            total += v1[i] * v2[i];
	return total;
}

TYPE
rnd_range(TYPE from, TYPE to)
{
    double r;
    drand48_r(&rng_buf, &r);
    return (TYPE) (r * (from - to) + to);
}

void
set_vals_serial(TYPE v[], int n, TYPE lo, TYPE hi, long seed)
{
    int i;
    if (rng_init == 0) {
        srand48_r(seed, &rng_buf);
        rng_init = 1;
    }
    for (i = 0; i < n; i++)
        v[i] =  rnd_range(lo, hi);
}

void
set_vals_parallel(TYPE v[], int n, TYPE lo, TYPE hi, long seed)
{
    int i;

    #pragma omp parallel
    {
        if (rng_init == 0) {
            srand48_r(seed + omp_get_thread_num() * SEED_MULT, &rng_buf);
            rng_init = 1;
        }
        #pragma omp for
        for (i = 0; i < n; i++)
            v[i] =  rnd_range(lo, hi);
    }
}

void
set_vals_non_rnd(TYPE v[], int n, TYPE lo, TYPE hi, long seed)
{
    for (int i = 0; i < n; i++)
            v[i] =  (TYPE) i + 1;
}


long double
sum_vec(TYPE v[], int n)
{
    long double total = 0.0;
    for (int i = 0; i < n; i++)
        total += v[i];
    return total;
}

int
    main(int argc, char *argv[])
{
    int n = 10, iterations = 5;
    bool verbose = false;
    long seed = time(NULL);
    time_t start, end;
    TYPE *v1, *v2;
    double result;

    process_args(argc, argv, &n, &iterations, &seed, &verbose);

    printf("Executing with n = %d and for %d iterations.\n", n, iterations);

    v1 = (TYPE *) malloc(sizeof(TYPE) * n);
    if (v1 == NULL) {
        fprintf(stderr, "Can't allocate v1.\n");
        exit(EXIT_FAILURE);
    }
    v2 = (TYPE *) malloc(sizeof(TYPE) * n);
    if (v1 == NULL) {
        fprintf(stderr, "Can't allocate v2.\n");
        exit(EXIT_FAILURE);
    }

    set_vals_parallel(v1, n, LO, HI, seed);
    set_vals_parallel(v2, n, LO, HI, seed);

    time(&start);
    puts("Starting one core");
    for (int i = 0; i < iterations; i++)
        result = dotprod_one_core(v1, v2, n);
    time(&end);
    printf("Result: %f\n", result);
    printf("Time taken: %ld %ld %ld\n", start, end, end - start);
    puts("Finishing one core");

    time(&start);
    puts("Starting multicore");
    for (int i = 0; i < iterations; i++)
        result = dotprod_multicore(v1, v2, n);
    time(&end);
    printf("Result: %f\n", result);
    printf("Time taken: %ld %ld %ld\n", start, end, end - start);
    puts("Finishing multicore");

    time(&start);
    puts("Starting target");
    for (int i = 0; i < iterations; i++)
        result = dotprod_target(v1, v2, n);
    time(&end);
    printf("Result: %f\n", result);
    printf("Time taken: %ld %ld %ld\n", start, end, end - start);
    puts("Finishing target");

    if (verbose) {
        puts("v1");
        print_vector(v1, n);
        puts("v2");
        print_vector(v2, n);
    }
    free(v1);
    free(v2);

    return EXIT_SUCCESS;
}



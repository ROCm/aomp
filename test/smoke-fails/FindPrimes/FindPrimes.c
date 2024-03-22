
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdbool.h>
#include <math.h>
#include <omp.h>

int compare(const void* a, const void* b) {
    return (*(int*)a - *(int*)b);
}

int count_primes(int* primes, int n, int i1, int i2, int* output) {
    int count = 0;
    int sqrt_i2 = (int)sqrt(i2);

    // Find the number of primes up to and equal to the square root of i2
    int num_primes = 0;
    while (num_primes < n && primes[num_primes] <= sqrt_i2) {
        num_primes++;
    }

    // Check each number in the interval
#pragma omp target map(to: primes[0:num_primes]) map(from: output[0:(i2-i1)/2]) map(tofrom: count)
#pragma omp parallel for num_threads(256)
    for (int i = i1; i <= i2; i++) {
        int is_prime = 1;

        // Check if i is divisible by any of the primes
        for (int j = 0; j < num_primes; j++) {
            if (i % primes[j] == 0) {
                is_prime = 0;
                break;
            }
        }

        // If i is prime, add it to the output array and increment the count
        if (is_prime) {
#pragma omp critical
            {
                output[count] = i;
                count++;
            }
        }
    }
    return count;
}


int main() {

    // Seeding the prime array with the primes from 2^1 to 2^2.
    int* primes = (int*)malloc(2 * sizeof(int));
    primes[0] = 2;
    primes[1] = 3;

    int n_primes = 2;
    int count = 0;
    for (int n = 2; n <= 22; n++) {
        int i1 = pow(2, n);
        int i2 = pow(2, n + 1);
        int* output = (int*)malloc((i2 - i1) / 2 * sizeof(int));
        count = count_primes(primes, n_primes, i1, i2, output);
        
        primes = (int*)realloc(primes, (n_primes + count) * sizeof(int));
        
        memcpy(primes + n_primes, output, count * sizeof(int));
        
        (void)qsort(primes, n_primes, sizeof(int), compare);

        n_primes = n_primes + count;
        printf("Number of primes between %d and %d: %d\n", i1, i2, count);
        free(output);
    }

    if (count == 268216) {
        return 0;
    }
    else {
        return -1;
    }
}


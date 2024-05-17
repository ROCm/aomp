
#include <stdio.h>
#include <stdbool.h>
#include <math.h>

bool is_prime(int n) {
    if (n < 2) {
        return false;
    }
    for (int i = 2; i <= sqrt(n); i++) {
        if (n % i == 0) {
            return false;
        }
    }
    return true;
}

#include <stdbool.h>
#include <math.h>

bool is_primeT(int p, int* table, int size) {
    if (p < 2) {
        return false;
    }
    for (int i = 0; i < size; i++) {
        if (table[i] * table[i] > p) {
            break;
        }
        if (p % table[i] == 0) {
            // printf("false %d %d\n", p, table[i]);
            return false;
        }
    }
    return true;
}

int main() {

#define n 1000

    int primes[n];
    int count_gpu = 0;
    int count_cpu = 0;
    int count = 0;
    int num = 2;
    while (count < n) {
        if (is_prime(num)) {
            primes[count] = num;
            count++;
        }
        num++;
    }

    printf("Table of the first %d prime numbers:\n", n);
    for (int i = 0; i < n; i++) {
        printf("%d ", primes[i]);
    }
    printf("\n");

    int last_prime = primes[n - 1];
    int last_prime_square = last_prime * last_prime;

    printf("Primes from %d to %d:\n", last_prime, last_prime_square);

    count_gpu = 0;
#pragma omp target teams distribute parallel for map(to:primes[0:n]) reduction(+:count_gpu)
    for (int i = last_prime; i <= last_prime_square; i = i + 2) {
        if (is_primeT(i, primes, n)) {
            count_gpu++;
        }
    }
    printf("OpenMP Offloading Count = %d \n", count_gpu);

    count_cpu = 0;
    for (int i = last_prime; i <= last_prime_square; i = i + 2) {
        if (is_primeT(i, primes, n)) count_cpu++;
    }
    printf("CPU Count = %d , These two numbers should be equal.\n", count_cpu);

    if (count_cpu == count_gpu) {
        return 0;
    }
    else {
        return -1;
    }
}

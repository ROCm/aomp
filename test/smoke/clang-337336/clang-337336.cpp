// Copyright © Advanced Micro Devices, Inc., or its affiliates.
//
// SPDX-License-Identifier:  MIT

#include <assert.h>
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <cmath>

#define GB (1024 * 1024 * 1024)
const size_t SIZE = 3 * 1024L * 1024L;
const unsigned int NUM_TEAMS = 3;

int main(int argc, char * argv[])
{
    unsigned int NKERNELS = 10;
    if (argc > 1){
      float memSize = atof(argv[1]);
      float sizeGiB;
      float reservedMemPercent = 10;
      for (int i = NKERNELS; i > 0; i--) {
        sizeGiB = (i * (double)SIZE * sizeof(double) / GB);
        printf("sizeGiB: %f, memSize: %f, reservedMemPercent: %f\n", sizeGiB, memSize, reservedMemPercent);
        if (sizeGiB < (memSize * (1 - reservedMemPercent/100))) {
          NKERNELS = i;
          break;
       }
    }
      printf("NKERNELS reduced to: %d\n", NKERNELS);
    }
    int dummy = 0;

    printf("Total Memory = %.6lf GB\n\n", NKERNELS * (double)SIZE * sizeof(double) / GB);

#pragma omp target
    {
        dummy += 1;
    }
    {
        double *p[NKERNELS];
        unsigned const check = 9;

        for (unsigned int i = 0; i < NKERNELS; ++i)
        {
            p[i] = new double[SIZE];
            assert(p[i] != nullptr);
            for (size_t j = 0; j < SIZE; ++j)
            {
                p[i][j] = 2.0;
            }
#pragma omp target enter data map(to:p[:NKERNELS]) 
#pragma omp target enter data map(to:p[i][:SIZE])

        }

        double start = omp_get_wtime();
        for (unsigned j = 0; j < NKERNELS; j++)
        {
#pragma omp target teams distribute parallel for num_teams(NUM_TEAMS)
            for (size_t i = 0; i < SIZE; ++i)
            {
                p[j][i] += sqrt((double)i);
                p[j][i] *= log(p[j][i]);
            }
        }
        double end = omp_get_wtime();
        printf("Time for %u kernels - blocking = %.8lf Sec. \n", NKERNELS, end - start);
        for (unsigned j = 0; j < NKERNELS; j++)
        {
#pragma omp target update from(p[j][:SIZE])
            printf("p[%u][%u] = %.8lf \n", j, check, p[j][check]);
        }

        for (unsigned int i = 0; i < NKERNELS; ++i)
        {
            for (size_t j = 0; j < SIZE; ++j)
            {
                p[i][j] = 2.0;
            }
#pragma omp target update to(p[i][:SIZE])
        }

        start = omp_get_wtime();
        for (unsigned j = 0; j < NKERNELS; j++)
        {
#pragma omp target teams distribute parallel for nowait num_teams(NUM_TEAMS)
            for (size_t i = 0; i < SIZE; ++i)
            {
                p[j][i] += sqrt((double)i);
                p[j][i] *= log(p[j][i]);
            }
        }
#pragma omp taskwait
        end = omp_get_wtime();
        printf("\nTime for %u kernels - nowait = %.8lf Sec. \n", NKERNELS, end - start);
        for (unsigned j = 0; j < NKERNELS; j++)
        {
#pragma omp target update from(p[j][:SIZE])
            printf("p[%u][%u] = %.8lf \n", j, check, p[j][check]);
        }
        for (unsigned int i = 0; i < NKERNELS; ++i)
        {
            delete p[i];
        }
    }
    return 0;
}

#include <stdio.h>
#include <stdlib.h>

int hello_(int *chkval, int *mult)
{
    int val = *chkval;
    val *= *mult;
    printf("hello hst %d*%d = %d\n", *chkval, *mult, val);
    fflush(stdout);
    return val;
}

#pragma omp declare target

int hello_dev_(int *chkval, int *mult)
{
    int val = *chkval;
    val *= *mult;
    printf("hello dev %d*%d = %d\n", *chkval, *mult, val);
    return val;
}

#pragma omp end declare target

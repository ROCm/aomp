#include <stdio.h>
#include <omp.h>

void findNeighborsDispl( const int *localPadding , int *res)
{

    int stack[64];
    int stackptr = 0;
    stack[stackptr++] = -1;
    int child = 0;
    int node = 0;

    do
    {
      if(localPadding[child] > 0)
        stack[stackptr++] = child;
        node = stack[--stackptr];
    } while (node > 0);
    *res = node;
}


void findNeighbors(const int *localPadding, int maz, int may, int max, int *res)
{
    for (int hz = 0; hz <= maz; hz++)
        for (int hy = 0; hy <= may; hy++)
            for (int hx = 0; hx <= max; hx++)
                findNeighborsDispl(localPadding, res);
}

int main() {
    const int maz = 1;
    const int may = 2;
    const int max = 3;
    int res;
    int o_localPadding[1000];
#pragma omp target teams distribute parallel for map(to: o_localPadding[0:1000],maz, may, max)

    for (int pi = 0; pi < 10; pi++)
    {
        findNeighbors(o_localPadding, max, may, maz, &res);

    }
    return 0;
}


#include <stdio.h>
#include <omp.h>

int main(){
    int a[256];
    int b[256];
    int c;
    int d;
    int e[256];
    int f;

#pragma omp target teams distribute parallel for map(tofrom: a)
for (int i =0; i <256; i++)
{
    a[i] =i;
}

#pragma omp target teams distribute map(tofrom: b)
for(int i = 0; i <256; i++)
{
    b[i]= i;
}

#pragma omp target parallel map(tofrom: c)
{
    c = 2;
}
#pragma omp target teams map(tofrom: d)
{
   d = 3;
}

#pragma omp target parallel for map(tofrom: e)
for (int i =0; i <256; i++)
{
    e[i] =i;
}

#pragma omp target map(tofrom: f)
{
    f = 4;
}
if (f != 4) return 1;
return 0;
}



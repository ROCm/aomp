#include <stdio.h>
#include <stddef.h>
#include <time.h>
#include <malloc.h>
void foobar(int iter) {
struct timespec t0,t1;
clock_gettime(CLOCK_REALTIME, &t0);
double x0 = 1.0;
double sumx = 0.0;
double n = 1000.0;
int bump = 0;
int ArrSize = 1000000;
double *arr = (double *)malloc((ArrSize+bump)*8);
//double arr[100+bump];
#pragma omp target enter data map(to:x0, arr[0:ArrSize+bump]) device(0)
#pragma omp target teams distribute parallel for map(tofrom: sumx) reduction(+:sumx) device(0)
for (int i = 0; i < n; ++i ) {
sumx += x0;
}
#pragma omp target exit data map(delete: x0, arr[0:ArrSize+bump]) device(0)
if(sumx != n){
fprintf(stderr, "Fail\n");
return;
}
else
if (iter %40 == 0) {
fprintf(stderr, "Success! %d\n",iter);
clock_gettime(CLOCK_REALTIME, &t1);
double m = (t1.tv_sec - t0.tv_sec) + (t1.tv_nsec - t0.tv_nsec)/1e9;
fprintf(stderr, "Time %g for 1000 loops\n", m);
t0 = t1;
bump++;
}
free (arr);
}

int main(int argc, char ** argv) {
for (int iter=0; iter < 40*5; iter++) 
  foobar(iter);
}

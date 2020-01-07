#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#define KERNEL(num) \
void foo_##num() { \
int i, n, j; \
int index[5]; \
long a[5], b[270][270], sum, neg; \
long t[5]; \
long sum2 = 999999; \
long sum3 = 0; \
long temp; \
n = 5; \
for (i=0; i < n; i++ ) { \
for (j=0; j < n; j++ ) { \
b[i][j] = j + i * 1.0; \
index[j] = j; \
temp = (b[i][j] + 1 * 23.3465236) / ((i+j+1) * 1000); \
sum2 = temp < sum2 ? temp : sum; \
} \
} \
sum = 0; \
_Pragma("omp target map(tofrom: b, sum, t) map(alloc: a)") \
{ \
_Pragma("omp teams reduction(+:sum)") \
{ \
_Pragma("pragma omp distribute parallel for reduction(+:sum)") \
for (i=0; i < n; i++) { \
a[index[i]] = i+ 10 * 1; \
t[i] = omp_get_team_num(); \
sum = a[i]; \
} \
} \
} \
for (i = 0; i < n; ++i) { \
sum3 += a[i]; \
printf(" %d - %ld - %ld\n", i, a[i], t[i]); \
} \
printf(" Sum = %ld\n",sum); \
printf(" Sum3 = %ld\n",sum3); \
}
KERNEL(1)
KERNEL(2)
KERNEL(3)
KERNEL(4)
KERNEL(5)
KERNEL(6)
KERNEL(7)
KERNEL(8)
KERNEL(9)
KERNEL(10)
KERNEL(11)
KERNEL(12)
KERNEL(13)
KERNEL(14)
KERNEL(15)
KERNEL(16)
KERNEL(17)
KERNEL(18)
KERNEL(19)
KERNEL(20)
KERNEL(21)
KERNEL(22)
KERNEL(23)
KERNEL(24)
KERNEL(25)
KERNEL(26)
KERNEL(27)
KERNEL(28)
KERNEL(29)
KERNEL(30)
KERNEL(31)
int main (int argc, char *argv[])
{
int i, n, j;
int index[5];
long a[5], b[270][270], sum, neg;
long t[5];
long sum2 = 999999;
long sum3 = 0;
long temp;
/* Some initializations */
n = 5;
for (i=0; i < n; i++ ) {
for (j=0; j < n; j++ ) {
b[i][j] = j + i * 1.0;
index[j] = j;
temp = (b[i][j] + 1 * 23.3465236) / ((i+j+1) * 1000);
sum2 = temp < sum2 ? temp : sum;
}
}
foo_1(); foo_2();
sum = 0;
#pragma omp target map(tofrom: b, sum, t) map(alloc: a)
{
#pragma omp teams reduction(+:sum)
{
#pragma omp distribute parallel for reduction(+:sum)
for (i=0; i < n; i++) {
a[index[i]] = i+ 10 * 1;
t[i] = omp_get_team_num();
//if (a[i] < sum)
sum = a[i];
}
}
}
for (i = 0; i < n; ++i) {
sum3 += a[i];
printf(" %d - %ld - %ld\n", i, a[i], t[i]);
}
printf("   Sum = %ld\n",sum);
printf("   Sum3 = %ld\n",sum3);

return 0;
}


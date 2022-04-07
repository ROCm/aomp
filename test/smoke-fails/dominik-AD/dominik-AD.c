#include <stdio.h>

int main()
{
double x[10];
int i;
double *x_d;
double x1[10];

for (i = 0; i < 10; ++i) {
	x[i] = 1;
	x1[i] = 3;
}
x_d = x1;
printf("2nd arg x1 should be equal to 3rd %p  %p  %p \n",x, x1, x_d);
#pragma omp target  enter data map(to:x)
#pragma omp target map(tofrom:x_d)
{
	x_d = x;
        printf("x on device : %p\n", x);
}
printf("1st arg x = to host x, 3rd arg = to device x %p %p  %p \n",x, x1, x_d);
printf("x_d %lf %lf \n",x_d[0], x_d[1]);
for (i = 0; i < 10; ++i) {
	x[i] = 4;
}
printf("1st arg x should be equal to 3rd %p  %p  %p \n",x, x1, x_d);
printf("x_d %lf should equal to %lf \n",x_d[0], x[0]);
if (x_d[0] != x[0]) return 1;
return 0;
}

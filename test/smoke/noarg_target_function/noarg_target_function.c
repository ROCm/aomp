#include <stdio.h>
#pragma omp declare target
void my_noarg_func(){printf("This is a noarg function \n"); }
#pragma omp end declare target
int main(){
#pragma omp target
my_noarg_func();
return 0;
}

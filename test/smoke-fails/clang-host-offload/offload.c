/* compile with:



clang -g -O0 -fopenmp -fopenmp-targets=x86_64-unknown-linux-gnu -Xopenmp-target=x86_64-unknown-linux-gnu -march=znver1 -o offload offload.c



run with:



OMP_TARGET_OFFLOAD=mandatory ./offload



or:



./offload
*/
#include <stdio.h>
#include <stdlib.h>
#include <omp.h>



int main(int argc, char * argv[]) {
int on_device = 0;
#pragma omp target map(from:on_device)
{
on_device = !omp_is_initial_device();
}
printf("ran on device: %s\n", on_device ? "yes" : "no");
return EXIT_SUCCESS;
}

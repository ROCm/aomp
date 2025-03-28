#include <omp.h>
#include <stdio.h>
int main() {
  printf("omp_get_num_devices():%d  ROCR_VISIBLE_DEVICES:%s\n",
         omp_get_num_devices(), getenv("ROCR_VISIBLE_DEVICES"));
  return 0;
}

#include <omp.h>
#include <stdio.h>
int main() {
  printf("HOST:   initial_device:%d  default_device:%d   num_devices:%d\n",
         omp_get_initial_device(), omp_get_default_device(),
         omp_get_num_devices());
#pragma omp target teams
#pragma omp parallel
  if ((omp_get_thread_num() == 0) && (omp_get_team_num() == 0))
    printf("TARGET: device_num:%d  num_devices:%d get_initial:%d  "
           "is_initial:%d \n        threads:%d  teams:%d\n",
           omp_get_device_num(), omp_get_num_devices(),
           omp_get_initial_device(), omp_is_initial_device(),
           omp_get_num_threads(), omp_get_num_teams());
}

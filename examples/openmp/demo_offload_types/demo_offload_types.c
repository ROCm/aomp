#include <omp.h>
#include <stdio.h>

// --- OpenMP Offload Types Demo ---
// This program demonstrates the use of OpenMP offloading features.
// It prints information about the host and target devices, including
// the number of devices, threads, and teams available.

int main() {
  // Print host device information
  printf("   HOST:   initial_device:%d  default_device:%d   num_devices:%d\n",
         omp_get_initial_device(), omp_get_default_device(),
         omp_get_num_devices());

  // Offload to target device and print device information
#pragma omp target teams
#pragma omp parallel
  if ((omp_get_thread_num() == 0) && (omp_get_team_num() == 0)) {
    printf("   TARGET: device_num:%d  num_devices:%d get_initial:%d  "
           "is_initial:%d \n           threads:%d  teams:%d\n",
           omp_get_device_num(), omp_get_num_devices(),
           omp_get_initial_device(), omp_is_initial_device(),
           omp_get_num_threads(), omp_get_num_teams());
  }

  return 0;
}

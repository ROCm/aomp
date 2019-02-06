#include <stdio.h>
#include <omp.h>

int main() {
  int num_devs = omp_get_num_devices();
  for (int device_num = 0; device_num < num_devs ; device_num++) {
#pragma omp target device(device_num) nowait
#pragma omp teams num_teams(2) thread_limit(4)
#pragma omp parallel num_threads(2)
    {
      // need to pass the total device number to all devices, per module load
      int num_threads = omp_get_num_threads();
      int num_teams   = omp_get_num_teams();
      int num_devices = omp_get_num_devices(); // not legal in 4.5

      // need to pass the device id to the device starting the kernel
      int thread_id   = omp_get_thread_num();
      int team_id     = omp_get_team_num();
      int device_id   = omp_get_device_num();  // no API in omp 4.5

      // assume we have homogeneous devices
      int total_threads = num_devices * num_teams * num_threads;
      int gthread_id    = (device_id * num_teams * num_threads) + (team_id * num_threads) + thread_id;

      // print out id
      printf("Hello OpenMP 5 from \n");
      printf(" Device num  %d of %d devices\n", device_id, num_devices);
      printf(" Team num    %d of %d teams  \n", team_id,   num_teams);
      printf(" Thread num  %d of %d threads\n", thread_id, num_threads);
      printf(" Global thread %d of %d total threads\n", gthread_id, total_threads);
    };
  };
#pragma omp taskwait
  printf("The host device num is %d\n", omp_get_device_num());
  printf("The initial device num is %d\n", omp_get_initial_device());
  printf("The number of devices are %d\n", num_devs);
}


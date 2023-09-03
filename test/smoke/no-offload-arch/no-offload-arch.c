#include <stdio.h>
#include <omp.h>

int main(void) {
  int isHost = 1;

#pragma omp parallel for
  for (int i =0; i<5; i++) {
    printf("Hello world. iteration %d num_thread %d \n", i, omp_get_thread_num());
  }
#pragma omp target map(tofrom: isHost)
  {
    isHost = omp_is_initial_device();
    printf("Hello world. %d\n", 100);
    for (int i =0; i<5; i++) {
      printf("Hello world. iteration %d num_thread %d \n", i, omp_get_num_threads());
    }
  }

  printf("Target region executed on the %s\n", isHost ? "host" : "device");

  return isHost;
}


// helloworld.c
#include <stdio.h>
#include <omp.h>

int main(void) {
  int isHost = 1;

#pragma omp target map(tofrom: isHost)
  {
    isHost = omp_is_initial_device();
    printf("Hello world. %d\n", 100);
    for (int i =0; i<5; i++) {
      printf("Hello world. iteration %d\n", i);
    }
  }

  printf("Target region executed on the %s\n", isHost ? "host" : "device");

  return isHost;
}

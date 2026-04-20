#include <stdio.h>
#include <omp.h>

int main(void) {
  int isHost = 1;
  int host_val = 0;
  int* host_val_ptr = &host_val;
  int rc;

#pragma omp target map(tofrom: isHost) is_device_ptr(host_val_ptr)
  {
    isHost = omp_is_initial_device();
    printf("Hello world. %d\n", 100);
    for (int i =0; i<5; i++) {
      printf("Hello world. iteration %d\n", i);
    }
    printf("123456789%n should write 9 to host_val_ptr\n", host_val_ptr);
  }
  rc = isHost;
  if (host_val != 9)
    rc++ ;
  printf("Target region executed on the %s\n", isHost ? "host" : "device");
  printf("The value of host_val from device printf is %d\n",host_val);
  printf("12345678%n should write 8 to host_val_ptr\n", host_val_ptr);
  printf("The value of host_val from host printf is %d\n",host_val);
  if (host_val != 8)
    rc++ ;

  return rc;
}


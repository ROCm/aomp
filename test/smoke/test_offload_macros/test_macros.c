#include <stdio.h>
#include <omp.h>

#include <offload_macros.h>
#define _VALUE_TO_STRING(x) #x
#define _STRVALUE(x) _VALUE_TO_STRING(x)

int main(void) {
  int isHost = 1;
#pragma omp target map(tofrom: isHost)
{
   isHost = omp_is_initial_device();
#ifdef _DEVICE_ARCH
   printf("DEVICE ARCH:%s   GPU(STR):%s\n",
         _STRVALUE(_DEVICE_ARCH), _STRVALUE(_DEVICE_GPU));
#else
   printf("THIS IS AN UNEXPECTED HOST FALLBACK IN TARGET REGION!\n");
#endif
}
  printf("Target region executed on the %s\n", isHost ? "host" : "device");
  return isHost;
}

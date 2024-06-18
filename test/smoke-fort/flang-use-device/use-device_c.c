#include <assert.h>
#include <malloc.h>
#include <stdio.h>

int *get_ptr() {
  int *ptr = malloc(sizeof(int));
  assert(ptr && "malloc returned null");
  return ptr;
}

int check_result(int *host_ptr, int *dev_ptr) {
  // printf("Host_Ptr = %p\nDev_Ptr = %p\n", host_ptr, dev_ptr);
  if (dev_ptr == NULL || dev_ptr == host_ptr) {
    printf("FAILURE\n");
    return -1;
  } else {
    printf("SUCCESS\n");
    return 0;
  }
}
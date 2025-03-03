#include "omp.h"
#include <stdio.h>
#include <stdarg.h>

extern "C" int foo(int *buf, int count, int tag ) ;
extern "C" int bar(int *buf, int count, int source, int tag) ;

int main(int argc, char *argv[]) {
  int test_array[2];
  test_array[0] = 10;
  test_array[1] = 100;
  int host_foo_result = foo(test_array, 1, 2);
  int host_bar_result = bar(test_array, 1, 2, 3);
  printf("PRE-TARGET results  host_foo_result:%d   host_bar_result:%d\n\n", 
      host_foo_result,host_bar_result);

#pragma omp target
  {
    printf("TARGET START\n");
    int device_foo_result = foo(test_array, 1, 2);
    int device_bar_result = bar(test_array, 1, 2, 3);
    printf("TARGET results  device_foo_result:%d   device_bar_result:%d\n", 
      device_foo_result,device_bar_result);
    test_array[0] = 1000;
    test_array[1] = 10000;
    printf("TARGET END;  Setting test_array[0]=%d  test_array[1]=%d\n\n",test_array[0],test_array[1]);
  }

  int post_target_foo_result = foo(test_array, 1, 2);
  int post_target_bar_result = bar(test_array, 1, 2, 3);
  printf("POST TARGET results  foo result:%d   bar result:%d\n\n", 
      post_target_foo_result, post_target_bar_result);
  return 0;
}

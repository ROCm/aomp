/*
program main
    implicit none
    integer :: a = 10

    !$omp target data map(tofrom: a)

    !$omp target map(tofrom: a)
        a = a + 1
    !$omp end target

    !$omp target update from(a)

    !$omp end target data

    a = -15
    !$omp target update from(a)

    print *, a  !<-- expect -15; actual 11
end program main


2) segfault
$>cat t.f

program main
    implicit none
    integer :: a = 10

    !$omp target map(tofrom: a)
        a = a + 1
    !$omp end target

    !$omp target update from(a)

    a = -15
    !$omp target update from(a)

    print *, a  !<-- expect -15; actual segfault
end program main

 */

#include <stdio.h>

#define TEST1 1
#define TEST2 1

int main()
{
  int a;  

  // test 1
#if TEST1
  a = 10;
  #pragma omp target data map(tofrom: a)
  {
    #pragma omp target map(tofrom: a)
    {
      a = a + 1;
    }
    //printf("test 1: a is %d (after target)\n", a);

    #pragma omp target update from(a)
  }
  //printf("test 1: a is %d (after target data)\n", a);

  a = -15;

  #pragma omp target update from(a)
  printf("test 1: a is %d\n", a);
#endif

#if TEST2
  // test 2
  a = 10;  
  #pragma omp target map(tofrom: a)
  {
    a = a + 1;
  }

  #pragma omp target update from(a)

  a = -15;

  #pragma omp target update from(a)
#endif

  printf("test 2: a is %d\n", a);

  return 1;
}

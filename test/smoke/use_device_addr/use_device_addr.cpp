#include <iostream>
#include <stdio.h>
#include <omp.h>

int main(int argc, char **argv){

  double data=10000;

  double *ptr1=nullptr,*ptr2=nullptr;
  
#pragma omp target enter data map(to:data,ptr1)


#pragma omp target map(tofrom: data,ptr1)
    ptr1 = &data;


#pragma omp target data use_device_addr(data)
    ptr2 = &data;

#pragma omp target exit data map(from:data,ptr1)
  

  if (ptr1==ptr2){
    printf("use_device_addr test :: Pass \n");
    return 0;
  }
  else{
    printf("use_device_addr test :: FAIL \n Addresses are :: %p and %p \n",ptr1, ptr2);
    return 1;
  }
}

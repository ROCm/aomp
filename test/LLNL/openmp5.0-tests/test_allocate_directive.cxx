#include <iostream>
#include "omp.h"

int main(int argc, char **argv){

  const int N=300;
  double data[N];

  const double a=1.0,b=2.0,c=3.0;
  // Notes This doesnt compile
  // Will should a different d per team. Which d will be copied back
  // if at all ?
  // How to check if data is in const memory ?
  //#pragma omp allocate(a,b,c) allocator(omp_const_mem_alloc)
  double d=0.0;
  //#pragma omp allocate(d) allocator(omp_pteam_mem_alloc)
  
#pragma omp target enter data map(to: data)
  
#pragma omp target map(tofrom: data[N/3:2*N/3])
#pragma omp teams distribute parallel for 
  for( int i=0;i<N;i++){
    data[i]=a*i*i+b*i+c;
  }

#pragma omp target teams distribute parallel for reduction(+:d)
for( int i=0;i<N;i++){
  d+=(data[i]-a*i*i-c)/b;
 }

#pragma omp target exit data map(from: data) map(from:d)

  int sum = 0.0;
  
  for( int i=0;i<N;i++){
    sum+= (data[i]-a*i*i-c)/b;
  }

  
  if (sum!=(N*(N-1)/2)){
    std::cout<<"Concurrent map test:: FAIL\n";
    std::cout<<"Sum is "<<sum<<"!="<<(N*(N-1)/2)<<"\n";
    return 1;
  }
  else {
    std::cout<<"Concurrent test:: Pass\n";
    std::cout<<sum<<" "<<d<<"\n";
    return 0;
  }
}

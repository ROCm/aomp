#include <iostream>

#pragma omp requires unified_shared_memory

int main(int argc, char **argv){

const int N=10000;
#ifdef STATIC
 double data[N]; // This works with clang
#else
 double *data;   // This fails with clang
 data = new double[N];
#endif

 for( int i=0;i<N;i++) data[i]=0;
 
#pragma omp target teams distribute parallel for 
 for( int i=0;i<N;i++){
   data[i]=i*i+i;
 }
 
 
 int sum = 0.0;
 
 for( int i=0;i<N;i++){
   sum+= data[i]-i*i;
 }
 
 
 if (sum!=(N*(N-1)/2)){
   std::cout<<"Requires unified shared memory test:: FAIL\n";
   std::cout<<"Sum is "<<sum<<"!="<<(N*(N-1)/2)<<"\n";
   return 1;
 }
 else {
   std::cout<<"Requires unifed shared memory test:: Pass\n";
   return 0;
 }
}

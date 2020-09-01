#include <iostream>

#include "omp.h"
#include <cuda_runtime.h>
size_t freemem();
int main(int argc, char **argv){
  
  const int N=10000;
  double data[N]; // This works with clang

  ssize_t last;
  ssize_t curr=freemem();

  std::cout<<"Free mem start "<<curr<<"\n";
 for( int i=0;i<N;i++) data[i]=0;

 static double a=1.0,b=2.0,c=3.0;
#pragma omp target enter data map(to: data)
 


 //#pragma omp target data map(threadprivate:a,b,c)
#pragma omp threadprivate(a,b,c)
#pragma omp target teams distribute parallel for 
 for( int i=0;i<N;i++){
   a=a*i*i;
   b=b*i;
   c=c+a+b;
   
   data[i]=c;
 }

 last = curr;
 curr = freemem();
 std::cout<<"Free mem post map "<<curr<<" delta = "<<(curr-last)<<"\n";
 if (!omp_pause_resource(omp_pause_soft,0)){
   std::cout<<" Soft omp_pause_resource FAILED\n";
 } else std::cout<< "Soft omp_pause_resource succeeded \n";

 last = curr;
 curr = freemem();
 std::cout<<"Free mem post pause "<<curr<<" delta = "<<(curr-last)<<"\n";
 if (last==curr) std::cout<<"Soft pause did not release any resources\n";

#pragma omp target exit data map(from: data)

 last = curr;
 curr = freemem();
 std::cout<<"Free mem post exit data  "<<curr<<" delta = "<<(curr-last)<<"\n";

 if (!omp_pause_resource(omp_pause_hard,0)){
   std::cout<<" Hard omp_pause_resource FAILED\n";
 } else std::cout<< "Hard omp_pause_resource succeeded \n";


 last = curr;
 curr = freemem();
 std::cout<<"Free mem post HARD pause "<<curr<<" delta = "<<(curr-last)<<"\n";
 if (last==curr) std::cout<<"Hard pause did not release any resources\n";



 

 
 
 int sum = 0.0;
 
 for( int i=0;i<N;i++){
   sum+= (data[i]-a*i*i-c)/b;
 }
 
 
 if (sum!=(N*(N-1)/2)){
   std::cout<<"Pause resources data integrity test:: FAIL\n";
   std::cout<<"Sum is "<<sum<<"!="<<(N*(N-1)/2)<<"\n";
   return 1;
 }
 else {
   std::cout<<"Pause resources data integrity test:: Pass\n";
   return 0;
 }
}
size_t freemem(){
  size_t mfree,mtotal;
  if (cudaMemGetInfo(&mfree,&mtotal)!=cudaSuccess){
    std::cerr<<"cudaMemGetInfo FAILED\n";
  } else {
    return mfree;
  }
  return 1;
}

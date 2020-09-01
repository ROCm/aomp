#include <iostream>

#define N 100

void cpu_foo(double *, double *b, double *c);
void gpu_foo(double *, double *b, double *c);

#pragma omp declare variant( cpu_foo ) match( construct={parallel} )
#pragma omp declare variant( gpu_foo ) match( construct={target}   )
void foo( double *a, double *b, double *c){
  for (int i=0;i<N;i++) c[i]+= a[i]+ b[i];
}

void cpu_foo( double *a, double *b, double *c){
#pragma omp parallel for
  for (int i=0;i<N;i++) c[i]+= a[i]+ b[i];
}

#pragma omp declare target
void gpu_foo( double *a, double *b, double *c){
#pragma omp teams distribute parallel for
for (int i=0;i<N;i++) c[i]+= a[i]+ b[i];
}
#pragma omp end declare target
  
int main(int argc, char **argv){

double a[N],b[N],c[N];
  
for(int i=0;i<N;i++){
  a[i]=1.0;
  b[i]=2.0;
  c[i]=i;
 }
 

 
#pragma omp parallel
 {
   foo(a,b,c);
 }


 int sum = 0.0;
 
 for(int i=0;i<N;i++) sum+= c[i];
 
 if (sum!=((N-1)*N/2+3*N)){
   std::cout<<" Declare variant part 1 :: FAIL\n";
   std::cout<< " Sum "<<sum<<" != "<< ((N-1)*N/2+3*N)<<"\n";
   return 1;
 } else {
   std::cout<<" Declare variant part 1 :: Pass \n";
 }
 
 for (int i=0;i<N;i++) c[i]=i;
 

 
 
 
 
 
#pragma omp target map(to: a[:N],b[:N]) map (tofrom: c[:N])
 {
   foo(a,b,c);
 }
 
 


 if (sum!=((N-1)*N/2+3*N)){
   std::cout<<" Declare variant part 2 :: FAIL\n";
   std::cout<< " Sum "<<sum<<" != "<< ((N-1)*N/2+3*N)<<"\n";
   return 1;
 } else {
   std::cout<<" Declare variant part 2 :: Pass \n";
 }
}

#include <iostream>

int main(int argc, char **argv){

  struct str { 
    
    int* d;
  } s;
    
s.d = new int[100];
  for(int i=0;i<100;i++) s.d[i]=0;

#pragma omp target data map(tofrom:s)
#pragma omp target teams distribute parallel for
  for(int i=0;i<100;i++) s.d[i]=i*i+i;

#pragma omp target data map(tofrom:s.d[0:100])
#pragma omp target teams distribute parallel for
  for(int i=0;i<100;i++) s.d[i]+=1;

#pragma omp target data map(tofrom:s) map(tofrom:s.d[0:100])
#pragma omp target teams distribute parallel for
  for(int i=0;i<100;i++) s.d[i]+=1;


  int sum=0;
  for(int i=0;i<100;i++) sum += s.d[i]-i*i-2;
  if (sum!=(99*100)/2){
    std::cout<<"Pointer attach test: FAIL\n"<<sum<<"!="<<50*99<<"\n";
    for(int i=0;i<100;i++) std::cout<<i<<" "<<s.d[i]-i*i-2<<"\n"<<std::flush;
    return 1;
  } else {
    std::cout<<"Pointer attach test: PASS\n";
    return 0;
  }
    
 delete [] s.d;
}

